"""Pre-flight check before pushing this repository to GitHub.

    python tools/preflight.py

Fails (exit 1) if the repository is not safe/legal to publish:
  * any file bigger than 1 MB (model weights must never be committed)
  * any checkpoint-like file (.pth/.pt/.safetensors/.bin/.onnx)
  * a missing LICENSE / LICENSE_ZH.txt / NOTICE / README.md
  * a .py file that does not compile
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 1024 * 1024
FORBIDDEN_EXT = {".pth", ".pt", ".safetensors", ".bin", ".onnx", ".gguf", ".ckpt", ".h5"}
REQUIRED = ["README.md", "LICENSE", "LICENSE_ZH.txt", "NOTICE", "apply.py", "vram.py"]
SKIP_DIRS = {".git", "__pycache__", "node_modules"}


def main():
    problems = []
    total = 0
    count = 0
    py_files = []

    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = os.path.join(base, name)
            size = os.path.getsize(path)
            rel = os.path.relpath(path, ROOT)
            total += size
            count += 1
            ext = os.path.splitext(name)[1].lower()
            if ext in FORBIDDEN_EXT:
                problems.append(f"model/checkpoint file must not be committed: {rel}")
            if size > MAX_BYTES:
                problems.append(f"file larger than 1 MB: {rel} ({size / 1024 ** 2:.1f} MB)")
            if ext == ".py":
                py_files.append(path)

    for name in REQUIRED:
        if not os.path.isfile(os.path.join(ROOT, name)):
            problems.append(f"required file missing: {name}")

    import py_compile

    for path in py_files:
        try:
            py_compile.compile(path, doraise=True)
        except Exception as exc:
            problems.append(f"syntax error in {os.path.relpath(path, ROOT)}: {exc}")

    print(f"{count} files, {total / 1024:.1f} KB total, {len(py_files)} python files")

    notice = os.path.join(ROOT, "NOTICE")
    if os.path.isfile(notice):
        with open(notice, "r", encoding="utf-8") as handle:
            if "4.1(a)" not in handle.read():
                problems.append("NOTICE does not carry the license 4.1(a) disclaimer")

    if problems:
        print("\nFAILED:")
        for item in problems:
            print("  - " + item)
        return 1

    print("\nOK: safe to publish (no weights, license + disclaimer in place, syntax clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
