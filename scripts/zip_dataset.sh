#!/usr/bin/env bash
# Bundle data/raw/ into data.zip for upload to Google Drive.
#
# Usage:
#   ./scripts/zip_dataset.sh [output.zip]
#   ./scripts/zip_dataset.sh --view <name> [output.zip]
#
# With --view, bundles data/processed/<name>/ (built by build_trainset.py)
# instead of data/raw/. A view is symlinks into data/raw; zip follows them and
# stores real file contents, so the archive is self-contained.
#
# Default output is ./data.zip in the repo root. After running, upload the zip
# to MyDrive/coin_sorter/data.zip (matches the path the Colab notebook reads).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/data/raw"

if [[ "${1:-}" == "--view" ]]; then
    [[ -n "${2:-}" ]] || { echo "error: --view needs a name" >&2; exit 1; }
    SRC_DIR="$REPO_ROOT/data/processed/$2"
    [[ -d "$SRC_DIR" ]] || {
        echo "error: $SRC_DIR missing. Build it first:" >&2
        echo "  python scripts/build_trainset.py --view $2" >&2
        exit 1
    }
    shift 2
fi

OUT_ZIP="${1:-$REPO_ROOT/data.zip}"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "error: source dir $SRC_DIR does not exist" >&2
    exit 1
fi

# Refuse to bundle an empty dataset. A view is symlinks, hence -type l too;
# zip follows them and stores the real contents (that is zip's default; -y
# would store the links themselves, which Colab could not read).
if ! find "$SRC_DIR" -mindepth 2 \( -type f -o -type l \) -print -quit | grep -q .; then
    echo "error: $SRC_DIR contains no images. Run capture.py first." >&2
    exit 1
fi

echo "Bundling $SRC_DIR -> $OUT_ZIP"
# Zip from inside data/raw so class folders sit at the top level of the archive
# (this is what the Colab notebook expects).
# _rejected/ holds quarantined crops (prune_crops.py) and must not train.
( cd "$SRC_DIR" && zip -qr "$OUT_ZIP" . -x "*.gitkeep" -x "_rejected/*" -x "_calib/*" )

SIZE=$(du -h "$OUT_ZIP" | cut -f1)
echo "Done: $OUT_ZIP ($SIZE)"
echo
echo "Next steps:"
echo "  1. Upload $OUT_ZIP to: MyDrive/coin_sorter/data.zip"
echo "  2. Open notebooks/train_colab.ipynb in Colab and Run All."
