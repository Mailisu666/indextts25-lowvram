"""Installer for the IndexTTS-2.5 low-VRAM patch set.

    python apply.py --root "D:\\path\\to\\IndexTTS2.5-package"

What it does
------------
1. copies ``vram.py`` next to ``indextts/infer_v2_5.py``
2. patches ``indextts/infer_v2_5.py`` (capability-based dtype + stage swapping)
3. patches ``webui.py`` (``--dtype`` and ``--lowvram`` flags)
4. writes ``*.bak_pre_lowvram`` backups next to every file it touches

Every edit is anchored on exact source text and must match exactly once, so a
different upstream revision fails loudly instead of being half-patched.  The
script is idempotent: running it twice is a no-op.

Nothing here touches model weights, so it is safe to run on an existing
installation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

INFER = os.path.join("indextts", "infer_v2_5.py")
WEBUI = "webui.py"
VRAM_TARGET = os.path.join("indextts", "vram.py")

DTYPE_NEW = (
    "        # Resolve the compute type for this GPU (see indextts/vram.py). A plain\n"
    "        # torch.cuda.is_bf16_supported() probe is not enough: it reports True on\n"
    "        # pre-Ampere cards through emulation, which measured ~1.9x SLOWER than fp32\n"
    "        # on an RTX 2060 while still using only half the memory.\n"
    "        self.half_requested = bool(self.use_bf16)\n"
    '        half_mode = pick_half_dtype(self.device, verbose=True) if self.half_requested else "fp32"\n'
    '        self.use_bf16 = half_mode == "bf16"\n'
    '        self.use_fp16 = half_mode == "fp16"\n'
    '        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(half_mode)\n'
)

GPT_CAST_OLD = "        if self.use_bf16:\n            self.gpt.eval().bfloat16()\n        else:\n            self.gpt.eval()\n"
GPT_CAST_NEW = (
    "        if self.use_bf16:\n"
    "            self.gpt.eval().bfloat16()\n"
    "        elif self.use_fp16:\n"
    "            self.gpt.eval().half()\n"
    "        else:\n"
    "            self.gpt.eval()\n"
)

INFER_PATCHES = [
    {
        "name": "import vram helper",
        "olds": ["from indextts.utils.checkpoint import load_checkpoint\n"],
        "new": "from indextts.utils.checkpoint import load_checkpoint\nfrom indextts.vram import ModuleParking, pick_half_dtype, pick_lowvram_tier\n",
    },
    {
        "name": "dtype by capability",
        # pristine upstream, or the same file after an earlier manual fp16 tweak
        "olds": [
            "        self.dtype = torch.bfloat16 if self.use_bf16 else None\n",
            DTYPE_NEW,
        ],
        "new": DTYPE_NEW,
    },
    {
        "name": "gpt cast for fp16",
        "olds": [GPT_CAST_OLD, GPT_CAST_NEW],
        "new": GPT_CAST_NEW,
    },
    {
        "name": "kv cache dtype",
        "olds": [
            "        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_bf16)\n",
        ],
        "new": "        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_bf16 or self.use_fp16)\n",
    },
    {
        "name": "configurable chunk size",
        "olds": [
            "        if self.low_vram and not stream_return and len(text) > 40:\n"
            "            segments = self.split_text_by_punctuation(text, max_chars=40)\n",
        ],
        "new": "        # Peak VRAM scales with the audio length of one chunk, not with the total\n"
        "        # text length. Small cards can lower this to trade prosody for headroom.\n"
        '        chunk_chars = int(os.environ.get("INDEXTTS_CHUNK_CHARS", "40") or 40)\n'
        "        if self.low_vram and not stream_return and len(text) > chunk_chars:\n"
        "            segments = self.split_text_by_punctuation(text, max_chars=chunk_chars)\n",
    },
    {
        "name": "swap gpt on fp32 cards",
        "olds": [
            "        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_bf16 or self.use_fp16)\n",
        ],
        "new": "        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_bf16 or self.use_fp16)\n"
        "        if self.dtype is None:\n"
        "            # fp32 path (pre-Turing, e.g. GTX 1060): the backbone alone is ~3.1 GB,\n"
        "            # so it steps aside too while the reference encoder runs.\n"
        '            self._parking.register_swapped("gpt", self.gpt)\n',
    },
    {
        "name": "low-vram tier + parking manager",
        "olds": [
            '            if total_vram_gb < 10.0:\n'
            "                self.low_vram = True\n"
            '                print(f">> Low-VRAM mode enabled ({total_vram_gb:.1f} GB < 10 GB), long text will be split into chunks")\n',
        ],
        "new": '            if total_vram_gb < 10.0:\n'
        "                self.low_vram = True\n"
        '                print(f">> Low-VRAM mode enabled ({total_vram_gb:.1f} GB < 10 GB), long text will be split into chunks")\n'
        "\n"
        "        # Decide what may stay resident. Below 8 GB the prompt encoders live in\n"
        "        # system RAM and only visit the GPU when conditioning is (re)computed.\n"
        "        self.vram_tier = pick_lowvram_tier(self.device, verbose=True)\n"
        "        self._parking = ModuleParking(self.device, tier=self.vram_tier)\n",
    },
    {
        "name": "park w2v-bert",
        "olds": [
            "        self.semantic_model = self.semantic_model.to(self.device)\n"
            "        self.semantic_model.eval()\n",
        ],
        "new": "        self.semantic_model = self.semantic_model.to(self._parking.park_device)\n"
        "        self.semantic_model.eval()\n"
        '        self._parking.register_parked("w2v_bert", self.semantic_model)\n',
    },
    {
        "name": "swap semantic codec",
        "olds": [
            "        self.semantic_codec = self.semantic_codec.to(self.device)\n"
            "        print('>> semantic_codec weights restored cost: ', time.perf_counter() - start_time)\n",
        ],
        "new": "        self.semantic_codec = self.semantic_codec.to(self.device)\n"
        "        print('>> semantic_codec weights restored cost: ', time.perf_counter() - start_time)\n"
        '        self._parking.register_swapped("semantic_codec", self.semantic_codec)\n',
    },
    {
        "name": "swap cfm",
        "olds": [
            "        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)\n",
        ],
        "new": "        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)\n"
        "        if not self.use_torch_compile:\n"
        "            # torch.compile bakes its own graphs; keep the module resident then.\n"
        '            self._parking.register_swapped("cfm", self.s2mel.models["cfm"])\n',
    },
    {
        "name": "swap bigvgan",
        "olds": [
            "        self.bigvgan.remove_weight_norm()\n        self.bigvgan.eval()\n",
        ],
        "new": "        self.bigvgan.remove_weight_norm()\n"
        "        self.bigvgan.eval()\n"
        '        self._parking.register_swapped("bigvgan", self.bigvgan)\n',
    },
    {
        "name": "park campplus",
        "olds": [
            "        self.campplus_model = campplus_model.to(self.device)\n"
            "        self.campplus_model.eval()\n",
        ],
        "new": "        self.campplus_model = campplus_model.to(self._parking.park_device)\n"
        "        self.campplus_model.eval()\n"
        '        self._parking.register_parked("campplus", self.campplus_model)\n',
    },
    {
        "name": "park after load",
        "olds": [
            "        # 进度引用显示（可选）\n"
            "        self.gr_progress = None\n"
            '        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None\n',
        ],
        "new": "        # 进度引用显示（可选）\n"
        "        self.gr_progress = None\n"
        '        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None\n'
        "\n"
        "        # Park the prompt encoders last, once every weight has reached the GPU.\n"
        "        self._parking.start()\n"
        '        self._parking.report("after load")\n',
    },
    {
        "name": "activate on conditioning miss",
        "olds": [
            "        # 如果参考音频改变了，才需要重新生成, 提升速度\n"
            "        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:\n",
        ],
        "new": "        # The prompt encoders may be parked in system RAM; pull them back only when\n"
        "        # the conditioning really has to be recomputed (a cache hit costs 0 bytes).\n"
        "        _need_spk = self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt\n"
        "        _need_emo = self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt\n"
        "        if _need_spk or _need_emo:\n"
        "            self._parking.activate()\n"
        "\n"
        "        # 如果参考音频改变了，才需要重新生成, 提升速度\n"
        "        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:\n",
    },
    {
        "name": "release after conditioning",
        "olds": [
            "        else:\n            emo_cond_emb = self.cache_emo_cond\n",
        ],
        "new": "        else:\n"
        "            emo_cond_emb = self.cache_emo_cond\n"
        "\n"
        "        if _need_spk or _need_emo:\n"
        "            self._parking.release()\n",
    },
]

WEBUI_PATCHES = [
    {
        "name": "cli flags",
        "olds": [
            'parser.add_argument("--qwen_emo", action="store_true", default=False, help="Load QwenEmotion even on a low-VRAM GPU, where it is skipped by default")\n',
        ],
        "new": 'parser.add_argument("--qwen_emo", action="store_true", default=False, help="Load QwenEmotion even on a low-VRAM GPU, where it is skipped by default")\n'
        'parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "off", "fp32", "fp16", "bf16"], help="Compute type for half precision inference")\n'
        'parser.add_argument("--lowvram", type=str, default="auto", choices=["auto", "off"], help="Stage-swap weights to system RAM on small cards")\n',
    },
    {
        "name": "apply cli flags",
        "olds": ["cmd_args = parser.parse_args()\n"],
        "new": "cmd_args = parser.parse_args()\n"
        'os.environ.setdefault("INDEXTTS_DTYPE", cmd_args.dtype)\n'
        'os.environ.setdefault("INDEXTTS_LOWVRAM", cmd_args.lowvram)\n',
    },
]


def patch_file(path, patches, dry_run=False):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    crlf = "\r\n" in text
    applied = 0
    for patch in patches:
        name = patch["name"]
        candidates = [
            (old.replace("\n", "\r\n") if crlf else old, patch["new"].replace("\n", "\r\n") if crlf else patch["new"])
            for old in patch["olds"]
        ]
        # Idempotency first: many patches extend their own anchor, so the patched
        # text contains the original one.
        if any(new != old and new in text for old, new in candidates):
            print(f"  skip {name} (already applied)")
            continue
        for old, new in candidates:
            if text.count(old) == 1:
                text = text.replace(old, new)
                print(f"  ok   {name}")
                applied += 1
                break
        else:
            raise SystemExit(f"ERROR [{os.path.basename(path)}] '{name}': anchor not found - was this file modified upstream?")
    if applied and not dry_run:
        backup = path + ".bak_pre_lowvram"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"  backup -> {backup}")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    return applied


def main():
    parser = argparse.ArgumentParser(description="Install the IndexTTS-2.5 low-VRAM patch set.")
    parser.add_argument("--root", required=True, help="IndexTTS-2.5 package directory")
    parser.add_argument("--dry-run", action="store_true", help="check anchors without writing")
    parser.add_argument("--no-launcher", action="store_true", help="do not copy the launcher into the package")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: not a directory: {root}")
    print(f"root: {root}")

    source_vram = os.path.join(HERE, "vram.py")
    if not os.path.isfile(source_vram):
        raise SystemExit("ERROR: vram.py is missing next to apply.py")
    target_vram = os.path.join(root, VRAM_TARGET)
    if not args.dry_run:
        shutil.copy2(source_vram, target_vram)
    print(f"  ok   copy {VRAM_TARGET}")

    total = 0
    for relative, patches in ((INFER, INFER_PATCHES), (WEBUI, WEBUI_PATCHES)):
        target = os.path.join(root, relative)
        if not os.path.isfile(target):
            raise SystemExit(f"ERROR: not found: {target}")
        print(f"patching {relative}")
        total += patch_file(target, patches, dry_run=args.dry_run)

    launcher = os.path.join(HERE, "launchers", "webui_6g.bat")
    if not args.no_launcher and os.path.isfile(launcher):
        shutil.copy2(launcher, os.path.join(root, "webui_6g.bat"))
        print("  ok   copy webui_6g.bat")

    if args.dry_run:
        print(f"dry run: {total} patch(es) would be applied")
        return

    import py_compile

    for relative in (INFER, WEBUI, VRAM_TARGET):
        py_compile.compile(os.path.join(root, relative), doraise=True)
    print(f"done: {total} patch(es) applied, py_compile OK")
    print("launch with: webui_6g.bat")


if __name__ == "__main__":
    sys.exit(main())
