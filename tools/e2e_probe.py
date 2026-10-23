"""Reproduce the README numbers on your own machine.

    python tools/e2e_probe.py --root . --lowvram auto
    python tools/e2e_probe.py --root . --lowvram off
    python tools/e2e_probe.py --root . --lowvram auto --dtype fp32

It builds the model, generates the same short sentence twice and prints peak
VRAM (torch + whole card) and RTF for each run.  Nothing is written outside
``--out`` (default: the current directory).
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="IndexTTS-2.5 package directory")
    parser.add_argument("--lowvram", default="auto", choices=["auto", "off"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "off", "fp32", "fp16", "bf16"])
    parser.add_argument("--text", default="今天天气不错，我们出去走走吧。")
    parser.add_argument("--ref", default=os.path.join("examples", "voice_01.wav"))
    parser.add_argument("--lang", default="ZH")
    parser.add_argument("--seg-tokens", type=int, default=60)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--out", default="probe_out.wav")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    os.environ["INDEXTTS_LOWVRAM"] = args.lowvram
    os.environ["INDEXTTS_DTYPE"] = args.dtype
    sys.path.insert(0, root)
    os.chdir(root)

    print(f"=== lowvram={args.lowvram} dtype={args.dtype} ===", flush=True)
    t0 = time.perf_counter()
    import torch  # noqa: E402
    from indextts.infer_v2_5 import IndexTTS2  # noqa: E402
    from indextts import vram as vrammod  # noqa: E402

    print("import %.1fs" % (time.perf_counter() - t0), flush=True)
    print("gpu:", vrammod.describe_gpu("cuda:0"), flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    total, free = vrammod.vram_stats("cuda:0")
    print(f"before load: {total - free:.2f}/{total:.2f} GB used", flush=True)

    t0 = time.perf_counter()
    tts = IndexTTS2(
        cfg_path=os.path.join(root, "checkpoints", "config.yaml"),
        model_dir=os.path.join(root, "checkpoints"),
        use_bf16=True,
        use_cuda_kernel=True,
        use_qwen_emo=False,
    )
    print("load %.1fs" % (time.perf_counter() - t0), flush=True)
    if hasattr(tts, "_parking"):
        tts._parking.report("after load")

    ref = args.ref if os.path.isabs(args.ref) else os.path.join(root, args.ref)
    out = os.path.abspath(args.out)
    for run in range(1, args.runs + 1):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tts.infer(
            spk_audio_prompt=ref,
            text=args.text,
            output_path=out,
            lang=args.lang,
            max_text_tokens_per_segment=args.seg_tokens,
        )
        elapsed = time.perf_counter() - start
        duration = None
        try:
            import torchaudio

            wav, sr = torchaudio.load(out)
            duration = wav.shape[-1] / sr
        except Exception as exc:  # pragma: no cover
            print("duration probe failed:", exc, flush=True)
        extra = "" if duration is None else f" audio={duration:.2f}s RTF={elapsed / duration:.2f}"
        print(f">>> run{run}: gen={elapsed:.2f}s{extra}", flush=True)
        if hasattr(tts, "_parking"):
            tts._parking.report(f"run{run}")

    print("PEAK torch allocated: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024 ** 3), flush=True)


if __name__ == "__main__":
    main()
