"""Low-VRAM support for IndexTTS-2.5 on 4-10 GB consumer GPUs.

Why this exists
---------------
``torch.cuda.is_bf16_supported()`` returns True on pre-Ampere cards because
PyTorch can *emulate* bf16.  On an RTX 2060 (SM 7.5, no BF16 ALUs) that
emulation measured 50.7 ms for a 4096x4096 GEMM, versus 27.0 ms in fp32 and
7.2 ms in fp16 -- i.e. "half precision" was ~1.9x slower than full precision
while still only using half the memory.  IndexTTS-2.5 derives its ``use_bf16``
flag from that function, so 6 GB Turing cards silently take the slow path.

``pick_half_dtype()`` replaces that runtime probe with a capability rule:

    SM >= 8.0  Ampere/Ada/Hopper -> bf16  (native, wide exponent range)
    SM == 7.x  Turing/Volta      -> fp16  (Tensor cores on RTX 20xx, ~2x fp32
                                          through paired FP16 CUDA cores on GTX 16xx)
    SM < 7.0   Pascal/Maxwell    -> fp32  (FP16 runs at 1/64 rate on GP10x)

``ModuleParking`` then splits the pipeline into two mutually exclusive
residencies instead of loading every weight on the card at once:

  conditioning stage  ->  w2v-BERT (reference audio encoder, 2.2 GB) is on the
                          GPU, while the vocoder side (semantic codec, DiT/CFM,
                          BigVGAN) is parked in system RAM.

  generation stage    ->  the GPT backbone generates mel codes after the
                          conditioning is cached, so w2v-BERT goes back to RAM
                          and the vocoder side returns.

Because conditioning is cached per reference audio, the swap happens once per
prompt change - not once per text segment.  A 6 GB card drops from ~5.4 GB of
permanently resident weights to ~4.0 GB peak, which is the difference between
Windows silently spilling to shared memory (everything "works", 10x slower)
and living on the card.

Environment overrides:

    INDEXTTS_DTYPE    auto (default) | off | fp32 | fp16 | bf16
    INDEXTTS_LOWVRAM  auto (default) | off

Nothing here imports project modules, so it is safe to import from
``indextts.infer_v2_5`` at module load time.
"""

from __future__ import annotations

import os

import torch

__all__ = [
    "ModuleParking",
    "describe_gpu",
    "pick_half_dtype",
    "pick_lowvram_tier",
    "vram_stats",
]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if not value or not value.strip() else value.strip().lower()


def _cuda_index(device) -> int:
    text = str(device)
    return int(text.split(":")[-1]) if ":" in text else 0


def _is_cuda(device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def vram_stats(device="cuda:0"):
    """Return ``(total_gb, free_gb)`` for the physical card, or ``(None, None)``."""
    if not _is_cuda(device):
        return None, None
    index = _cuda_index(device)
    total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
    try:
        free = torch.cuda.mem_get_info(index)[0] / 1024 ** 3
    except Exception:  # pragma: no cover - very old drivers
        free = total - torch.cuda.memory_allocated(index) / 1024 ** 3
    return total, free


def describe_gpu(device="cuda:0") -> str:
    if not _is_cuda(device):
        return str(device)
    props = torch.cuda.get_device_properties(_cuda_index(device))
    return f"{props.name} (SM {props.major}.{props.minor}, {props.total_memory / 1024 ** 3:.1f} GB)"


def pick_half_dtype(device="cuda:0", verbose: bool = False) -> str:
    """Pick ``'bf16'``, ``'fp16'`` or ``'fp32'`` for half-precision inference."""
    forced = _env("INDEXTTS_DTYPE", "auto")
    capability = None
    if _is_cuda(device):
        capability = torch.cuda.get_device_capability(_cuda_index(device))

    if forced in ("off", "fp32"):
        choice, reason = "fp32", "disabled by INDEXTTS_DTYPE"
    elif forced in ("fp16", "bf16"):
        choice, reason = forced, "forced by INDEXTTS_DTYPE"
    elif capability is None:
        if str(device).startswith("xpu"):
            choice, reason = "bf16", "XPU device"
        else:
            choice, reason = "fp32", "no CUDA device"
    elif capability[0] >= 8:
        choice, reason = "bf16", "native BF16 support"
    elif capability[0] == 7:
        choice, reason = "fp16", "Turing/Volta: BF16 is emulated, FP16 is fast"
    else:
        choice, reason = "fp32", "pre-Turing: FP16 runs at a reduced rate, BF16 is emulated"

    if verbose:
        print(f">> half precision: {choice} ({reason})")
    return choice


def pick_lowvram_tier(device="cuda:0", verbose: bool = False) -> str:
    """``'off'`` keeps everything resident; ``'ref'`` alternates the two stages."""
    if _env("INDEXTTS_LOWVRAM", "auto") == "off":
        return "off"
    total, _ = vram_stats(device)
    if total is None:
        return "off"
    tier = "ref" if total < 8.0 else "off"
    if verbose and tier == "ref":
        print(f">> low-VRAM plan: stage swapping enabled ({total:.1f} GB card)")
    return tier


class ModuleParking:
    """Two mutually exclusive residencies instead of everything on the GPU.

    ``register_parked``    modules live in system RAM and visit the GPU while
                           conditioning is computed (w2v-BERT, campplus).
    ``register_swapped``   modules live on the GPU but step aside during
                           conditioning, because that stage does not use them
                           (semantic codec, CFM/DiT, BigVGAN).
    """

    def __init__(self, device="cuda:0", tier: str = "off", verbose: bool = True):
        self.device = device
        self.tier = tier
        self.verbose = verbose
        self.enabled = tier != "off" and _is_cuda(device)
        self._parked = {}
        self._swapped = {}
        self._active = False

    @property
    def park_device(self):
        """Where a parked module should be created/loaded."""
        return "cpu" if self.enabled else self.device

    def register_parked(self, name: str, module):
        if module is not None:
            self._parked[name] = module
        return module

    def register_swapped(self, name: str, module):
        if module is not None:
            self._swapped[name] = module
        return module

    def _move(self, modules, target):
        for module in modules.values():
            module.to(target)

    def start(self) -> None:
        """Park the reference encoders once every weight has been loaded."""
        if not self.enabled:
            return
        self._move(self._parked, "cpu")
        torch.cuda.empty_cache()
        if self.verbose:
            parked = ", ".join(self._parked) or "-"
            swapped = ", ".join(self._swapped) or "-"
            print(f">> low-VRAM: parked in RAM [{parked}], stage-swapped [{swapped}]")

    def activate(self) -> None:
        """Enter the conditioning stage: reference encoders in, vocoder side out."""
        if not self.enabled or self._active:
            return
        self._move(self._parked, self.device)
        self._move(self._swapped, "cpu")
        self._active = True
        torch.cuda.empty_cache()
        if self.verbose:
            print(">> low-VRAM: conditioning stage (w2v-BERT on GPU, vocoder parked)")

    def release(self) -> None:
        """Return to the generation stage: reference encoders out, vocoder side in."""
        if not self.enabled or not self._active:
            return
        self._move(self._parked, "cpu")
        self._move(self._swapped, self.device)
        self._active = False
        torch.cuda.empty_cache()
        if self.verbose:
            print(">> low-VRAM: generation stage (vocoder on GPU, w2v-BERT parked)")

    def report(self, tag: str = "") -> None:
        if not _is_cuda(self.device):
            return
        index = _cuda_index(self.device)
        total, free = vram_stats(self.device)
        reserved = torch.cuda.memory_reserved(index) / 1024 ** 3
        peak = torch.cuda.max_memory_allocated(index) / 1024 ** 3
        print(
            f">> VRAM {tag}: {total - free:.2f}/{total:.2f} GB used, "
            f"torch reserved {reserved:.2f} GB, peak {peak:.2f} GB"
        )
