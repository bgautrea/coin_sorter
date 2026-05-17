#!/usr/bin/env bash
# Fetch a trained ONNX model into ./models/coin_classifier.onnx.
#
# Usage:
#   ./scripts/fetch_model.sh <source>
#
# Where <source> is one of:
#   - A local path:     /some/path/best.onnx
#   - An scp target:    user@host:/path/best.onnx
#   - An http(s) URL:   https://example.com/best.onnx
#
# For Google Drive: easiest is to scp from a workstation that has Drive
# mounted, or use `gdown` (install with `pip install gdown`).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/models"
DEST="$DEST_DIR/coin_classifier.onnx"
mkdir -p "$DEST_DIR"

if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
usage: $0 <source>

source forms:
  /local/path/model.onnx
  user@host:/remote/path/model.onnx
  https://example.com/model.onnx
EOF
    exit 1
fi

SRC="$1"

# Back up any existing model so a bad fetch doesn't silently overwrite a good one.
if [[ -f "$DEST" ]]; then
    BACKUP="$DEST.$(date +%Y%m%d_%H%M%S).bak"
    cp "$DEST" "$BACKUP"
    echo "Backed up existing model -> $BACKUP"
fi

case "$SRC" in
    http://*|https://*)
        echo "Downloading $SRC -> $DEST"
        curl -fL --progress-bar "$SRC" -o "$DEST"
        ;;
    *:*)
        echo "scp $SRC -> $DEST"
        scp "$SRC" "$DEST"
        ;;
    *)
        if [[ ! -f "$SRC" ]]; then
            echo "error: source file $SRC not found" >&2
            exit 1
        fi
        echo "cp $SRC -> $DEST"
        cp "$SRC" "$DEST"
        ;;
esac

SIZE=$(du -h "$DEST" | cut -f1)
echo "Done: $DEST ($SIZE)"
