#!/usr/bin/env python3
"""Install ultralytics' dependencies, minus the ones this image must keep.

ultralytics is installed --no-deps because letting pip resolve its tree on a
Jetson is dangerous: it would replace the Jetson-built torch/torchvision with
CPU-only PyPI wheels, add opencv-python alongside the image's
opencv-contrib-python, and pull numpy 2.x (which breaks the 1.26 ABI that
opencv and torch here are compiled against).

Hand-listing the rest does not work either -- it silently misses things that
only blow up mid-run, like polars, which ultralytics 8.4 imports when it saves
a checkpoint. So read the real requirement list from package metadata and skip
only the protected names.
"""
import importlib.metadata as md
import re
import subprocess
import sys

PROTECTED = {
    "torch", "torchvision", "torchaudio",   # Jetson CUDA builds
    "opencv-python", "opencv-python-headless",  # image ships opencv-contrib
    "numpy",                                 # 1.26 ABI, pinned by constraint
}

reqs = md.requires("ultralytics") or []
wanted = []
for raw in reqs:
    spec, _, marker = raw.partition(";")
    # Skip optional extras (export, dev, ...); we only need the base runtime.
    if "extra" in marker:
        continue
    name = re.split(r"[<>=!~\[\s]", spec.strip())[0].lower()
    if name in PROTECTED:
        print(f"  skip (protected): {spec.strip()}")
        continue
    wanted.append(spec.strip())

print("installing:", ", ".join(wanted) or "(nothing)")
if wanted:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "-c", "/tmp/constraints.txt", *wanted,
    ])
