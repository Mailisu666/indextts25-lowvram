"""Measure what a GPU actually does with fp32 / fp16 / bf16.

The whole point of indextts.vram.pick_half_dtype() is that
torch.cuda.is_bf16_supported() answers a different question than the one we
care about.  This script shows the difference on your own card:

    python tools/vram_bench.py

Interpreting the result
-----------------------
* fp16 much faster than fp32  -> Turing/Volta/Ampere with Tensor cores, or
                                 GTX 16xx through paired FP16 CUDA cores.
* fp16 much slower than fp32  -> Pascal/Maxwell (GTX 10xx, 9xx). Keep fp32.
* bf16 slower than fp32       -> BF16 is being *emulated*; do not use it.

Run it before reporting a speed problem: it tells you which compute type the
launcher should request for your card.
"""

from __future__ import annotations

import sys
import time

import torch


def bench_matmul(dtype, size=4096, iters=20, warmup=5):
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / iters
    return seconds, (2 * size ** 3) / seconds / 1e12


def main():
    if not torch.cuda.is_available():
        print("No CUDA device visible to torch.")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"GPU   {props.name} | SM {props.major}.{props.minor} | {props.total_memory / 1024 ** 3:.1f} GB")
    print(f"native BF16 (no emulation): {torch.cuda.is_bf16_supported(including_emulation=False)}")
    print(f"bf16_supported() (with emulation): {torch.cuda.is_bf16_supported()}")
    print()

    results = {}
    print(f"{'dtype':>10} {'ms/iter':>9} {'TFLOP/s':>9}")
    for name, dtype in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        try:
            seconds, tflops = bench_matmul(dtype)
        except Exception as exc:  # pragma: no cover - unsupported hardware
            print(f"{name:>10} {'failed':>9}   {exc}")
            continue
        results[name] = tflops
        print(f"{name:>10} {seconds * 1000:9.2f} {tflops:9.2f}")
        del dtype

    print()
    if "float32" not in results:
        return 1
    best = max(results, key=results.get)
    print(f"fastest: {best}")
    if results.get("float16", 0) > results["float32"] * 1.2:
        print("=> this card wants FP16 for half-precision inference")
    elif results.get("bfloat16", 0) >= results["float32"] * 0.9:
        print("=> this card has usable BF16; either BF16 or FP16 is fine")
    else:
        print("=> FP16 is not faster here and BF16 is emulated: stay on FP32")
    print("\nOverride any time with:  set INDEXTTS_DTYPE=fp32|fp16|bf16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
