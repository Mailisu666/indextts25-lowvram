# 实测数据与方法

所有数字都来自同一台机器、同一个整合包、同一段参考音频与文本，仅切换 `INDEXTTS_DTYPE` / `INDEXTTS_LOWVRAM`。

## 环境

| 项 | 值 |
|---|---|
| 显卡 | NVIDIA GeForce RTX 2060（6.00 GB，SM 7.5，无原生 BF16） |
| 系统 | Windows，驱动 616.92 |
| Python | 3.11.9（整合包自带 `python_embeded`） |
| PyTorch | 2.8.0+cu128 |
| 显存占用基线 | 加载前已有 1.01 GB 被系统/浏览器占用 |
| 参考音频 | `examples/voice_01.wav` |
| 文本 | 「今天天气不错，我们出去走走吧。」（16 字） |
| 分句上限 | 60 token |
| BigVGAN CUDA 内核 | 本次测试环境里加载失败（`PermissionError(13)`），自动回退到 PyTorch 实现 —— 三组配置一致，故对比有效；正常用户环境会更快一些 |

## 测法

- 显存："整卡占用" 取 `torch.cuda.mem_get_info()`（含系统与其他进程）；"torch 峰值"取 `torch.cuda.max_memory_allocated()`，每轮生成前 `reset_peak_memory_stats()`。
- 速度：`time.perf_counter()` 包住一次完整 `tts.infer(...)`；RTF = 耗时 / 生成音频时长。
- 每份配置跑两次生成：run1 含 CUDA 预热与首次换驻，run2 为稳态。稳定态数字取 run2。

复现脚本：[`tools/e2e_probe.py`](../tools/e2e_probe.py)

```bat
set INDEXTTS_LOWVRAM=auto
python_embeded\python.exe tools\e2e_probe.py auto
```

## 1) 原版路径：fp16 + 权重全部常驻（`INDEXTTS_LOWVRAM=off`）

```
>> half precision: fp16 (Turing/Volta: BF16 is emulated, FP16 is fast)
load 84.7s
>> VRAM after load: 6.00/6.00 GB used, torch reserved 5.03 GB, peak 4.98 GB
>>> run1: gen=31.42s audio=2.91s RTF=10.78
>> VRAM run1: 6.00/6.00 GB used, torch reserved 5.93 GB, peak 5.72 GB
>>> run2: gen=16.07s audio=3.55s RTF=4.52
>> VRAM run2: 6.00/6.00 GB used, torch reserved 5.93 GB, peak 5.73 GB
PEAK torch allocated: 5.73 GB
```

整卡占用在加载完成时就已经 **打满 6.00/6.00 GB**，生成阶段 `torch reserved` 涨到 5.93 GB —— 这就是 WDDM 溢出到共享内存的状态：不报错，但 RTF 4.52。

## 2) 本补丁：fp16 + 分阶段换驻（`auto`，6GB 卡自动启用）

```
>> half precision: fp16 (Turing/Volta: BF16 is emulated, FP16 is fast)
>> low-VRAM plan: stage swapping enabled (6.0 GB card)
>> low-VRAM: parked in RAM [w2v_bert, campplus], stage-swapped [semantic_codec, cfm, bigvgan]
>> VRAM after load: 4.29/6.00 GB used, torch reserved 3.28 GB, peak 3.47 GB
load 79.3s
>>> run1: gen=41.55s audio=3.39s RTF=12.26
>> VRAM run1: 5.05/6.00 GB used, torch reserved 4.00 GB, peak 4.94 GB
>>> run2: gen=4.49s audio=3.39s RTF=1.33
>> VRAM run2: 5.06/6.00 GB used, torch reserved 4.00 GB, peak 3.53 GB
PEAK torch allocated: 3.53 GB
```

峰值从 5.73 GB 降到 **3.53 GB**，稳态 RTF 从 4.52 降到 **1.33**。首次生成反而更慢（41.5 s vs 31.4 s），因为多付了两次阶段切换的 PCIe 传输。

## 3) fp32 路径：模拟 GTX 1060（`INDEXTTS_DTYPE=fp32` + 换驻）

Pascal 没有可用的 FP16 速率，只能走 fp32；此时 GPT 主干单独就是 3.1 GB，所以补丁把 GPT 也加入换驻集合。

```
>> half precision: fp32 (disabled by INDEXTTS_DTYPE)
>> low-VRAM: parked in RAM [w2v_bert, campplus], stage-swapped [gpt, semantic_codec, cfm, bigvgan]
>> VRAM after load: 5.36/6.00 GB used, torch reserved 4.35 GB, peak 4.32 GB
load 92.2s
>>> run1: gen=58.29s audio=3.42s RTF=17.02
>> VRAM run1: 6.00/6.00 GB used, torch reserved 5.40 GB, peak 6.48 GB
>>> run2: gen=4.79s audio=3.39s RTF=1.41
>> VRAM run2: 6.00/6.00 GB used, torch reserved 5.40 GB, peak 5.07 GB
PEAK torch allocated: 5.07 GB
```

稳态 RTF **1.41**，只比 fp16 慢 7%。注意 run1 的瞬时峰值达到 6.48 GB（超过物理显存，属于预热期的分配尖峰），所以**第一次生成会明显慢**，这是正常的。

> 这组是在 2060 上强制 fp32 跑的，**不等于 1060 实机**：1060 的 fp32 算力约 3.8 TFLOP/s（2060 是 5.1），实际会再慢一些，但显存路径与加载/换驻逻辑完全一致。

## 精度微基准（4096×4096 矩阵乘）

| 精度 | 每轮耗时 | 算力 |
|---|---|---|
| fp32 | 26.96 ms | 5.10 TFLOP/s |
| fp16 | 7.23 ms | 19.02 TFLOP/s |
| bf16 | 50.67 ms | 2.71 TFLOP/s |

`torch.cuda.is_bf16_supported()` → `True`；`torch.cuda.is_bf16_supported(including_emulation=False)` → `False`。

即：这张卡上的"bf16 支持"是软件模拟，GEMM 比 fp32 慢 1.9 倍。但端到端只差 7%（见上），因为生成耗时的大头是 CFM 的 25 步扩散与声码器，不是 GPT 的矩阵乘。

## 4) 长文本多分块压力测试（fp16 + 换驻）

85 字文本，按标点切成 3 块（`split_text_by_punctuation(max_chars=40)`），三块共用一个参考音频条件；
随后换一份参考音频（`voice_02`）再换回来，强制触发两轮 `activate/release`。

```
>> Low-VRAM: split into 3 segments: [...]
>> low-VRAM: conditioning stage (w2v-BERT on GPU, vocoder parked)
>> low-VRAM: generation stage (vocoder on GPU, w2v-BERT parked)
>>> voice_01:       gen=58.04s audio=17.44s RTF=3.33   peak 4.94 GB
>>> voice_02:       gen=21.74s audio=14.63s RTF=1.49   peak 5.13 GB
>>> voice_01-again: gen=21.80s audio=16.83s RTF=1.30   peak 5.14 GB
PEAK torch allocated: 5.14 GB
```

关键结论：

- 换驻**每个参考音频只发生一次**，不是每个分块一次（日志里 `conditioning/generation stage` 只出现一次）。
- 换参考音频后第一句会略慢（21.7 s vs 21.8 s 基本一致，说明切换开销可忽略）。
- **峰值显存随"单块音频时长"增长，而非文本总长**：3.4 s 音频 → 3.53 GB；约 6 s 一块 → 5.14 GB。
  这也是 `INDEXTTS_CHUNK_CHARS` 存在的原因：4GB 卡应下调到 20~25。

## 5) 真实 WebUI 启动验证（不是测试脚本，是 `webui.py` 本体）

用 `webui_6g.bat` 完整启动，确认 `webui.py` 的改动（`--dtype` / `--lowvram` / 环境变量）与补丁协同正常：

```
>> half precision: fp16 (Turing/Volta: BF16 is emulated, FP16 is fast)
>> low-VRAM plan: stage swapping enabled (6.0 GB card)
>> low-VRAM: parked in RAM [w2v_bert, campplus], stage-swapped [semantic_codec, cfm, bigvgan]
>> VRAM after load: 4.29/6.00 GB used, torch reserved 3.28 GB, peak 3.47 GB
* Running on local URL:  http://0.0.0.0:7860
```

`GET http://127.0.0.1:7860` → **HTTP 200**（176,503 bytes），此时整卡占用 **3,868 MiB**（3.78 GB）。

## 未覆盖

- GTX 1060 / 1660S 实机未验证。
- 未测试 `--torch_compile`、`--accel`、`--deepspeed` 与换驻的交互（前两者在多数整合包里不可用）。
- 未测试多卡、并发请求、超长文本（>1000 字）下的稳定性。
- 未通过 Gradio 界面按钮触发合成（用同一 `infer()` 入口以脚本调用验证）。
