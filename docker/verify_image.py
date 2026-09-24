#!/usr/bin/env python3
"""Fail the image build if anything on the training path is missing or wrong.

Checked at build time so a broken image cannot cost a training run. CUDA is
deliberately NOT checked here: `docker build` gets no GPU (daemon.json's
default-runtime applies to `docker run`, and BuildKit builds without device
access), so torch.cuda.is_available() is always False at build time.
train_jetson.sh --check tests it via `docker run`, where the answer is real.
"""
import importlib
import sys

REQUIRED = [
    ("numpy", "1."),        # opencv/torch here are built against the 1.26 ABI
    ("cv2", None),
    ("torch", None),
    ("torchvision", None),
    ("ultralytics", None),
    ("polars", None),       # ultralytics 8.3 reads results.csv with it on save
    ("matplotlib", None),
    ("scipy", None),
    ("PIL", None),
    ("yaml", None),
]

failed = []
for name, prefix in REQUIRED:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        failed.append(f"{name}: {type(e).__name__}: {e}")
        continue
    ver = getattr(mod, "__version__", "?")
    if prefix and not str(ver).startswith(prefix):
        failed.append(f"{name}: version {ver} does not start with {prefix!r}")
    else:
        print(f"  {name:<14} {ver}")

import torchvision  # noqa: E402  (already imported above if present)

if not ("a0" in torchvision.__version__ or "+" in torchvision.__version__):
    failed.append(
        f"torchvision {torchvision.__version__} looks like a stock PyPI build, "
        "not the Jetson CUDA one -- training would silently run on CPU"
    )

# The checkpoint-save path is where a missing transitive dep surfaces, an
# epoch into training. Exercise it now instead.
from ultralytics.utils import SETTINGS  # noqa: E402
print(f"  ultralytics settings loaded ({len(SETTINGS)} keys)")

if failed:
    print("\nIMAGE VERIFICATION FAILED:", file=sys.stderr)
    for f in failed:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("image verification OK")
