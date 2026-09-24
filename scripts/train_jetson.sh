#!/usr/bin/env bash
# Train a coin classifier on the Jetson Orin Nano, in a container.
#
# Why a container: you cannot `pip install torch` on a Jetson. PyPI wheels are
# x86 or CPU-only, and Ultralytics will happily pull one if it does not find a
# working torch -- giving you a trainer that runs at a few percent of speed
# with no error to tell you so. NVIDIA's aarch64+CUDA build ships inside the
# l4t-pytorch images, so we start from one and add ultralytics on top.
#
# Docker's default runtime on this box is already `nvidia`, so no --gpus flag
# is needed and the GPU is visible automatically.
#
# Usage (on the Jetson):
#   scripts/train_jetson.sh --check                  # verify CUDA only, no training
#   scripts/train_jetson.sh --rebuild --check        # force an image rebuild
#   scripts/train_jetson.sh --view denomination
#   IMAGE=dustynv/l4t-pytorch:r36.4.0 scripts/train_jetson.sh --view fine_v1
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-dustynv/l4t-pytorch:r36.4.0}"
# Derived image with ultralytics + base weights baked in, built on first use.
IMAGE="${IMAGE:-coin-sorter-train:r36.4.0}"
VIEW=""
CHECK_ONLY=0
REBUILD=0
# Orin Nano shares one pool of memory between CPU and GPU, so the stock
# batch=64 / workers=8 exhausts it. Override with BATCH=/WORKERS= if needed.
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --view)  VIEW="${2:?--view needs a name}"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        --rebuild) REBUILD=1; shift ;;
        *)       echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v docker >/dev/null; then
    echo "error: docker not found. This script is meant to run ON the Jetson." >&2
    exit 1
fi

# A Jetson reports its model here; guard against running this on the Pi.
if ! grep -qi tegra /proc/version 2>/dev/null; then
    echo "warning: this does not look like a Jetson (no 'tegra' in /proc/version)." >&2
fi

echo "==> image: $IMAGE (base $BASE_IMAGE)"
if [[ "$REBUILD" -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
    echo "==> building $IMAGE (one time; bakes in ultralytics and weights)"
    docker build --build-arg "BASE=$BASE_IMAGE" \
        -f "$REPO_ROOT/docker/Dockerfile.jetson" -t "$IMAGE" "$REPO_ROOT/docker"
fi

# --- CUDA sanity check ----------------------------------------------------
# Never start a long run without this. A CPU-only torch trains this dataset in
# hours instead of minutes and says nothing about it.
echo "==> checking CUDA inside the container"
docker run --rm -i "$IMAGE" python3 - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("FAIL: no torch in this image")
print("torch", torch.__version__)
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False -- wrong image for this JetPack")
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

[[ "$CHECK_ONLY" -eq 1 ]] && { echo "==> --check passed, stopping here."; exit 0; }
[[ -n "$VIEW" ]] || { echo "error: --view is required (or pass --check)" >&2; exit 2; }

DATA_DIR="$REPO_ROOT/data/processed/$VIEW"
[[ -d "$DATA_DIR" ]] || {
    echo "error: $DATA_DIR missing. Build it first:" >&2
    echo "  python scripts/build_trainset.py --view $VIEW" >&2
    exit 1
}

# The view is symlinks into data/raw, so data/raw must be mounted too or the
# links dangle inside the container. Mount the repo and keep paths identical.
TRAIN_N=$(find -L "$DATA_DIR/train" -type f 2>/dev/null | wc -l)
VAL_N=$(find -L "$DATA_DIR/val" -type f 2>/dev/null | wc -l)
if [[ "$TRAIN_N" -eq 0 || "$VAL_N" -eq 0 ]]; then
    echo "error: $DATA_DIR needs train/ and val/ subfolders (ultralytics" >&2
    echo "       check_cls_dataset requires them). Rebuild the view:" >&2
    echo "  python scripts/build_trainset.py --view $VIEW" >&2
    exit 1
fi
echo "==> training view '$VIEW' (train $TRAIN_N, val $VAL_N)"
# pin_memory allocates page-locked host RAM for a DMA transfer that does not
# exist when CPU and GPU share physical memory -- on Jetson it just burns the
# budget and cudaHostAlloc eventually fails with "CUDA error: out of memory".
echo "==> batch=$BATCH workers=$WORKERS pin_memory=off (unified memory)"
docker run --rm ${DOCKER_TTY:--i} \
    --ipc=host \
    -v "$REPO_ROOT:$REPO_ROOT" \
    -w "$REPO_ROOT" \
    -e "PYTHONPATH=$REPO_ROOT/src" \
    -e "PIN_MEMORY=False" \
    "$IMAGE" \
    bash -c "
        set -e
        # Ultralytics resolves a bare weights name against the working dir.
        [ -f yolo11n-cls.pt ] || cp /opt/weights/yolo11n-cls.pt .
        python3 -m coin_sorter.train --data '$DATA_DIR' --batch $BATCH --workers $WORKERS
    "

echo
echo "==> done. Copy the exported .onnx to models/coin_classifier.onnx and make"
echo "    sure classifier.labels matches the view's alphabetical class order:"
echo "      python scripts/build_trainset.py --view $VIEW --dry-run"
