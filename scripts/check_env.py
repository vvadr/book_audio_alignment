"""Check that the conda environment works on this machine.

Run from the project root:
    conda activate book-audio-alignment
    python scripts/check_env.py
"""

import importlib
import platform
import shutil
import subprocess
import sys

PACKAGES = ["torch", "transformers", "librosa", "soundfile", "silero_vad",
            "rapidfuzz", "pymupdf", "ipykernel"]


def main() -> int:
    ok = True
    print(f"python   {platform.python_version()}  ({sys.executable})")
    print(f"platform {platform.system()} {platform.machine()}")

    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            print(f"  {name:13} {version}")
        except Exception as error:  # report every failure, keep going
            ok = False
            print(f"  {name:13} FAILED: {type(error).__name__}: {error}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        first = subprocess.run([ffmpeg, "-version"], capture_output=True,
                               text=True).stdout.splitlines()[0]
        print(f"ffmpeg   {first.split(' Copyright')[0]}  ({ffmpeg})")
    else:
        ok = False
        print("ffmpeg   NOT FOUND on PATH (activate the conda env)")

    import torch

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    x = torch.ones(1024, 1024, device=device)
    value = (x @ x).sum().item()
    print(f"device   {device} ({name}), test matmul {'OK' if value == 1024 ** 3 else 'WRONG'}")
    if device == "cpu":
        print("WARNING: no GPU found, code will run on CPU")
    ok = ok and value == 1024 ** 3

    print("RESULT   OK" if ok else "RESULT   PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
