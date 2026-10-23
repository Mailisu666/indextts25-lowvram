# IndexTTS-2.5 低显存补丁（6GB 显卡实测可用）

让 **RTX 2060 / GTX 1660 Super / GTX 1060** 这类 6GB 显卡把 IndexTTS-2.5 跑稳，并且明显更快。

本仓库只包含**补丁与脚本**，不含模型权重、不含整合包的 Python 运行时。你需要一份已经能正常运行的 IndexTTS-2.5 环境（官方仓库或任意整合包）。

> **只想赶紧用上？** 三步：
> 1. 双击本仓库里的 **`INSTALL.cmd`**
> 2. 把你那个 IndexTTS 文件夹**拖进黑窗口**，回车
> 3. 双击 IndexTTS 目录里新出现的 **`webui_6g.bat`**，浏览器开 <http://127.0.0.1:7860>
>
> 装不上或想搞清楚原理，往下看「安装」章节。

---

## 实测结果

环境：RTX 2060 6GB / Windows / PyTorch 2.8.0+cu128 / IndexTTS-2.5 / 单卡单进程。

| 指标 | 原版（权重全部常驻） | 本补丁（分阶段换驻） |
|---|---|---|
| 加载完成后 torch 预留 | 5.03 GB | **3.28 GB** |
| 加载完成后整卡占用 | 6.00 / 6.00 GB（**已占满溢出**） | 4.29 / 6.00 GB |
| torch 峰值 | 5.73 GB | **3.53 GB** |
| 第二次生成（稳定态） | 16.07 s | **4.42 s** |
| RTF（耗时 / 音频时长，越低越快） | 4.52 | **1.30** |
| 首次生成（含预热） | 31.4 s | 41.5 s |

> 两次生成音频长度不同（3.55 s vs 3.39 s），因为采样是随机的（`do_sample=True`）；RTF 已按各自时长归一，可直接比较。
> **稳态生成快 3.5 倍，峰值显存降 2.2 GB。**

**加速到底来自哪里？** 主要不是精度，而是"不再溢出到共享内存"：

| 配置 | torch 峰值 | 稳态 RTF |
|---|---|---|
| fp16，权重全常驻（原版） | 5.73 GB | 4.52 |
| fp16，分阶段换驻 | 3.53 GB | **1.30** |
| fp32，分阶段换驻（1060 路径，在 2060 上强制 fp32 模拟） | 5.07 GB | **1.39** |

fp16 与 fp32 在稳态下只差 7%，因为耗时大头是 CFM 的 25 步扩散和声码器，不是 GPT 的矩阵乘。fp16 的真正价值是**同精度下显存最省**，并避开 bf16 模拟这条最差路径（GEMM 层面 bf16 模拟比 fp32 慢 1.9 倍）。

矩阵乘微基准（4096×4096，同一张 2060）：

| 精度 | 每轮耗时 | 算力 |
|---|---|---|
| fp32 | 26.96 ms | 5.10 TFLOP/s |
| **fp16** | **7.23 ms** | **19.02 TFLOP/s** |
| bf16 | 50.67 ms | 2.71 TFLOP/s |

> 注：`before load` 已有约 1.0 GB 被系统占用（桌面/浏览器），所以"整卡占用"比 torch 峰值高约 1.0~1.5 GB。

---

## 三个根因

### 1. bf16 在 Turing 上是"假支持"

`torch.cuda.is_bf16_supported()` 在 RTX 2060 上返回 **True**，但那是 **PyTorch 的软件模拟**（加 `including_emulation=False` 才是 `False`）。

实测模拟 bf16 的矩阵乘是 **50.67 ms**，比 fp32 的 27 ms **还慢 1.9 倍**，而 fp16 是 7.23 ms。

IndexTTS-2.5 的 `use_bf16` 正是从那个函数推出来的，所以 6GB Turing 卡上默认开的"半精度加速"实际上在拖后腿 —— 占了一半显存（这点是赚的），却比全精度还慢。

### 2. 权重全部常驻，超出 6GB 后 Windows 静默溢出

| 模块 | 大小 |
|---|---|
| GPT 主干 | 3.10 GB（fp32） |
| w2v-BERT 参考音频编码器 | 2.22 GB（fp32，**不随精度转换**） |
| semantic codec | 0.58 GB |
| BigVGAN | 0.43 GB |
| s2mel（含 DiT/CFM） | 0.40 GB |
| 合计 | **约 6.7 GB** |

6GB 卡放不下。Windows 的 WDDM 不会报 OOM，而是把超出部分丢到**共享内存**（走 PCIe），表现就是"能跑，但慢十倍" —— 这是最容易被误判为"显卡太弱"的情况。上表的原版列里，整卡占用 6.00/6.00 GB 就是这个状态。

### 3. 分句默认 120 token

中文 **1 字 = 1 token**（标点也算 1 个），所以默认一段约 117~120 个汉字。分句越长，单段激活显存越高。降到 60 后一段约 57~60 汉字（语言前缀 `<|zh|> ` 占掉几个 token）。

---

## 做了什么

三个独立机制，都可以单独关掉。

### `pick_half_dtype()`：按算力选精度，不信 `is_bf16_supported()`

| 算力 | 选择 | 原因 |
|---|---|---|
| SM ≥ 8.0（Ampere/Ada/Hopper） | bf16 | 原生支持，与上游行为一致 |
| SM 7.x（Turing/Volta） | fp16 | BF16 只能模拟；FP16 在 RTX 20 系走张量核心，在 GTX 16 系走成对 FP16 单元（约 2× fp32） |
| SM < 7.0（Pascal/Maxwell） | fp32 | GP10x 的 FP16 只有 1/64 速率，用 fp16 反而更慢 |

### `ModuleParking`：两套互斥驻留，而不是全部塞进显存

流水线天然分两段，两段用到的权重不重叠：

```
条件编码阶段（换参考音频时跑一次）      生成阶段（每个分句跑一次）
  w2v-BERT      2.2 GB  ← 在卡上          GPT 主干      ← 在卡上
  campplus      0.03 GB                   semantic codec
  ─────────────────────────               DiT/CFM
  声码器侧权重  → 暂存到内存               BigVGAN
```

- 因为上游本来就有参考音频条件缓存（`cache_spk_cond`），这套切换**每个参考音频只发生一次**，不是每个分句一次。
- 在 fp32 显卡（如 GTX 1060）上，GPT 主干单独就 3.1 GB，所以条件编码阶段它也会一起让位。

### 分句默认 60 token

启动脚本里带 `--gui_seg_tokens 60`。界面里的"分句最大Token数"滑杆仍可随时改。

---

## 支持矩阵

| 显卡 | 算力 | 自动精度 | 预期 | 实测 |
|---|---|---|---|---|
| RTX 2060 / 2060S / 2070 / 2080 | 7.5 + 张量核心 | fp16 | 舒适 | ✅ 峰值 3.53 GB，RTF 1.30 |
| GTX 1660 / 1660S / 1650 | 7.5 无张量核心 | fp16 | 可用 | ⚠️ 未实机验证 |
| GTX 1060 6G / 1080 | 6.1 | fp32 | 可用 | ✅ 以 `INDEXTTS_DTYPE=fp32` 在 2060 上模拟：峰值 5.07 GB，RTF 1.39；未实机验证 |
| RTX 3060 及以上 | 8.6+ | bf16 | 舒适 | — |
| 4GB 卡 | — | — | 建议同时降分句到 40、关闭 QwenEmotion | — |

"未实机验证"的项：判定逻辑是纯算力分支（不依赖具体型号），并且提供了 `tools/vram_bench.py` 让你在自己卡上直接测出该用哪种精度。

fp32 路径（1060 这类卡）额外做了一件事：`use_torch_compile` 之外，**GPT 主干也加入换驻集合** —— 因为 fp32 下它单独就占 3.1 GB，条件编码阶段用不到它。启动日志会显示：

```
>> low-VRAM: parked in RAM [w2v_bert, campplus], stage-swapped [gpt, semantic_codec, cfm, bigvgan]
```

---

## 安装

### 前提：你得先有一个能跑起来的 IndexTTS-2.5

本仓库**不含模型权重、不含 Python 运行时**，只有一个补丁。基础包请从下面任选一条路获取。

#### ① 整合包（Windows 新手最省事）

由 **FutureAI 实验室**制作维护，自带 `python_embeded`、汉化界面和启动脚本，解压即用：

- 整合包下载（UP 主视频简介里）：<https://www.bilibili.com/video/BV1Vibk6SERt/>

> 本补丁与 FutureAI 实验室无隶属关系，也不是其官方产物。整合包本体的著作权归其作者；本仓库**不转发**整合包（它 18.75 GB，且内含 NVIDIA CUDA 运行库，不在本仓库的授权范围内）。

#### ② 官方仓库（自己配环境，可复现）

```bash
git clone https://github.com/index-tts/index-tts
cd index-tts
# 按官方 README 装依赖（uv 或 pip 均可），CUDA 工具包建议 12.8+
```

下载权重（~10 GB，任选其一）：

```bash
# 方式 1：ModelScope（国内快）
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints

# 方式 2：HuggingFace
uv tool install "huggingface-hub"
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```

- 官方权重页：<https://modelscope.cn/models/IndexTeam/IndexTTS-2.5> ｜ <https://huggingface.co/IndexTeam/IndexTTS-2.5>
- 体积预期：主模型约 5.1 GB，辅助模型（w2v-BERT / BigVGAN / campplus / semantic codec）约 4.8 GB

#### 怎么确认手上这份是对的

打开那个文件夹，应该同时看到这三样：

```
webui.py
indextts\
checkpoints\          ← 里面有 gpt.pth / s2mel.pth / codec.pth，通常共 5~7 GB
```

并且**原版能正常启动过一次**（原版都跑不起来，打补丁没有意义）。

### 方式 A：双击安装（推荐，不用敲命令）

1. 把本仓库下载（Code → Download ZIP）或 `git clone` 到**任意位置**（不需要放进 IndexTTS 目录）
2. 双击仓库里的 **`INSTALL.cmd`**
3. 把上面那个 IndexTTS 文件夹**拖进黑窗口**，回车
4. 看到 `Done.` 后，双击 `<你的 IndexTTS 目录>\webui_6g.bat`
5. 浏览器打开 <http://127.0.0.1:7860>

`INSTALL.cmd` 会自动找 Python（优先包内自带的 `python_embeded`，其次系统里的），并检查目标目录是否含 `webui.py`、是否为 2.5 版本。**安装器本身不需要 torch**，任何 Python 3.8+ 都能跑。

### 方式 B：命令行

```bat
cd /d "解压出来的仓库目录"
python apply.py --root "D:\你的\IndexTTS2.5目录"
```

- 想先看会改什么、不实际写入：末尾加 `--dry-run`
- `python` 可以是任意 3.8+ 解释器，也可以是包内的 `python_embeded\python.exe`

脚本会做五件事：

1. 复制 `vram.py` 到 `<目录>\indextts\vram.py`
2. 给 `indextts/infer_v2_5.py` 打补丁（精度选择 + 分阶段换驻）
3. 给 `webui.py` 加 `--dtype` / `--lowvram` 参数
4. 每个被改动的文件旁边留一份 `*.bak_pre_lowvram` 备份
5. 复制启动器 `webui_6g.bat` 到包根目录

安全说明：

- **不改动任何模型权重**
- 每一处改动都以精确源码文本定位，必须**恰好匹配一次**，否则整体报错退出（不会打出半个补丁）
- **幂等**：重复运行是空操作
- 不改动的文件不会被碰

### 官方仓库用户注意

`webui_6g.bat` 是照整合包结构写的（调用 `python_embeded\python.exe`），官方仓库环境请手动启动：

```bash
uv run webui.py --version 2.5 --model_dir ./checkpoints_25 --fp16 --cuda_kernel --gui_seg_tokens 60
```

`--fp16` 只是"请求半精度"，最终用 fp16 还是 bf16 由补丁按显卡决定。

### 装好了怎么确认

启动日志里应出现这几行（数值随卡不同）：

```
>> half precision: fp16 (Turing/Volta: BF16 is emulated, FP16 is fast)
>> low-VRAM plan: stage swapping enabled (6.0 GB card)
>> low-VRAM: parked in RAM [w2v_bert, campplus], stage-swapped [semantic_codec, cfm, bigvgan]
>> VRAM after load: 4.29/6.00 GB used, torch reserved 3.28 GB, peak 3.47 GB
```

8GB 以上的卡**不会**出现 `low-VRAM plan` 那行 —— 这是正常的，说明换驻未启用（你要的就是原版满血路径）。

### 常见问题

| 现象 | 原因与解决 |
|---|---|
| `[ERROR] ...\webui.py not found` | 拖进去的不是包根目录。要选**直接包含 `webui.py` 的那一层**，不是它的上一层，也不是 `indextts` 子目录 |
| `[ERROR] ...\indextts\infer_v2_5.py not found` | 这是 IndexTTS 1.x 或 2.0，本补丁只支持 2.5 |
| `anchor not found - was this file modified upstream?` | 你的包版本与本补丁不同，或该文件已被手动改过。**此时没有任何文件被修改**，可用 `--dry-run` 看卡在哪一条 |
| 双击 `webui_6g.bat` 闪一下就没了 | 在 cmd 里手动运行它看报错。多半是包内没有 `python_embeded`（官方仓库环境请按上文手动启动） |
| 启动报 `indextts\vram.py` 找不到 | 安装器没跑成功，重跑 `INSTALL.cmd` |
| 卡在 `Running on local URL` 之前很久、控制台没输出 | 正常，首次加载 1~3 分钟，期间无输出 |
| 生成时 `CUDA out of memory` | 把 `INDEXTTS_CHUNK_CHARS` 调小（如 `25`），或关掉浏览器硬件加速等占显卡的程序 |
| 生成的音频有爆音/杂音 | 少数情况下 fp16 数值溢出。设 `INDEXTTS_DTYPE=fp32` 再试，或按下面回滚 |

### 回滚 / 卸载

在 IndexTTS 目录里执行：

```bat
copy /y "indextts\infer_v2_5.py.bak_pre_lowvram" "indextts\infer_v2_5.py"
copy /y "webui.py.bak_pre_lowvram" "webui.py"
del "indextts\vram.py"
del "webui_6g.bat"
```

等价于没装过；模型权重全程未被改动。

---

## 调参

| 环境变量 / 参数 | 默认 | 说明 |
|---|---|---|
| `INDEXTTS_DTYPE` | `auto` | `auto` / `off` / `fp32` / `fp16` / `bf16`；`off` 等于纯 fp32 |
| `INDEXTTS_LOWVRAM` | `auto` | `auto` / `off`；`off` 关闭分阶段换驻（8GB 以上卡自动就是 off） |
| `INDEXTTS_CHUNK_CHARS` | `40` | 低显存模式下按标点切块的字符数上限。**峰值显存与单块音频时长成正比**，4GB 卡建议设 `25` |
| `--dtype` / `--lowvram` | `auto` | 与上面两个环境变量等价，命令行优先 |
| `--gui_seg_tokens` | `60`（启动器） | 每段最大 token；中文 1 字 1 token |

### 长文本会怎样（实测）

85 字文本会被切成 3 块（每块约 40 字），三块共用一次条件编码：

| 场景 | 耗时 | 音频 | RTF | torch 峰值 |
|---|---|---|---|---|
| 首块（含预热） | 58.0 s | 17.4 s | 3.33 | 4.94 GB |
| 换参考音频后 | 21.7 s | 14.6 s | 1.49 | 5.13 GB |
| 换回原参考音频 | 21.8 s | 16.8 s | 1.30 | 5.14 GB |

结论：**峰值显存看的是"单块音频时长"，不是文本总长**。短句（3.4 s 音频）峰值 3.53 GB，40 字一块（约 6 s 音频）会到 5.14 GB。
6GB 卡余量约 1 GB；4GB 卡请把 `INDEXTTS_CHUNK_CHARS` 调到 `25` 甚至 `20`。

启动日志会明确告诉你选了什么、为什么：

```
>> half precision: fp16 (Turing/Volta: BF16 is emulated, FP16 is fast)
>> low-VRAM plan: stage swapping enabled (6.0 GB card)
>> low-VRAM: parked in RAM [w2v_bert, campplus], stage-swapped [semantic_codec, cfm, bigvgan]
>> VRAM after load: 4.29/6.00 GB used, torch reserved 3.28 GB, peak 3.47 GB
```

### 在自己的卡上验证该用哪种精度

```bat
python_embeded\python.exe tools\vram_bench.py
```

它会打印 fp32 / fp16 / bf16 的实测算力并给出建议。

---

## 已知限制

- **首次生成明显偏慢**（41 s vs 之后 4.4 s）：CUDA 内核预热 + 首次换驻。属正常现象。
- **`--torch_compile` 与分阶段换驻互斥**：编译后的图与权重搬移不兼容，补丁会自动跳过 CFM 的换出（`use_torch_compile` 时）。
- **`--accel` 不可用**：需要 `flash_attn`，多数整合包没装，会直接报错；本项目不涉及。
- **未实机验证 1060 / 1660S**：只有 2060 一张卡实测。1060 路径通过在 2060 上强制 `INDEXTTS_DTYPE=fp32` 模拟验证（见 `docs/RESULTS.md`）。
- **4GB 卡**：仍建议把分句降到 40，并用 `INDEXTTS_LOWVRAM=auto`；显存实在不够时可配合关闭其他占用 GPU 的程序（浏览器硬件加速）。
- 换参考音频会触发一次换驻，第一句会比后续略慢（约 0.3~1 s 的 PCIe 传输）。

---

## 许可与合规

- 模型与原始代码的版权归 **bilibili（index-tts 项目）** 所有，遵循随附的 `LICENSE` / `LICENSE_ZH.txt`（《bilibili 模型使用许可协议》）。
- 本仓库**只包含补丁与脚本**，不包含模型权重、不包含任何第三方整合包的运行时或启动器。
- 依据许可协议 3.4(b)，所有副本须保留原始版权声明与本许可协议 —— 本仓库已随附。
- 依据许可协议 4.1(a)，特此声明：

  > **该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、不担保、不承担责任。**

- 依 5.2，本仓库不使用 bilibili / IndexTTS 的商号或商标，仅作来源描述。
- 若你打算商用，请自行确认许可协议第 2.2 条的规模门槛与第 4.2 条的高风险场景限制。

## 致谢

- **[bilibili / index-tts](https://github.com/index-tts/index-tts)** —— IndexTTS-2.5 模型与原始代码的作者。
- **[FutureAI 实验室](https://space.bilibili.com/3546724261956069)** —— Windows 整合包的制作者，本补丁是在其整合包上验证并打磨的；整合包本体请从其发布渠道获取。
- **[NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)**、**[wenet](https://github.com/wenet-e2e/wenet)** 等上游项目 —— 声码器与说话人编码的实现来源。

本项目与 bilibili、index-tts 官方团队、以及任何整合包作者均无关联。
