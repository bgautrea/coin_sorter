#!/usr/bin/env python
"""Re-gate saved crops with the single-coin detector and quarantine the rest.

Crops saved before the single-coin gate existed (or with a loose one) include
touching/overlapping clusters, coins cut off at the edge and coins with a
neighbour in frame. Each crop is re-run through CoinDetector with thresholds
scaled for a tight crop (the coin should be ~40-50% of the image, centred,
clear of the border). Dry-run by default; --apply moves failures to
<raw_dir>/_rejected/<label>/ so they are excluded from training but not lost.

    python scripts/prune_crops.py            # report per label
    python scripts/prune_crops.py --apply    # actually move
    python scripts/prune_crops.py --sheet out.jpg   # contact sheet of rejects
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coin_sorter import load_config  # noqa: E402
from coin_sorter.capture import CoinDetector  # noqa: E402

# A tight crop pads the coin bbox by crop_pad_frac (0.25 -> coin is ~1/1.5^2
# = 44% of a square crop). Bracket that generously; the fill/border/isolation
# tests do the real work.
CROP_PARAMS = {
    "min_area_frac": 0.15, "max_area_frac": 0.75,
    "min_fill_frac": 0.85, "reject_border": True, "isolate": True,
    "crop_pad_frac": 0.0,  # isolation = any other blob anywhere in the crop
    "isolate_min_frac": 0.001,  # even a sliver of a neighbour at the edge counts
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--labels", nargs="*", help="only these labels (default: all)")
    p.add_argument("--apply", action="store_true", help="move rejects instead of just reporting")
    p.add_argument("--sheet", help="write a contact sheet of rejects to this path")
    args = p.parse_args()

    cfg = load_config(args.config)
    capc = cfg.get("capture", {})
    raw = Path(cfg["dataset"]["raw_dir"])
    params = {**capc, **CROP_PARAMS}
    labels = args.labels or sorted(d.name for d in raw.iterdir() if d.is_dir() and not d.name.startswith("_"))

    rejects: list[tuple[Path, str]] = []
    for label in labels:
        if label == capc.get("reject_label", "reject"):
            continue  # clusters/junk are the point of that class
        files = sorted((raw / label).glob("*.jpg"))
        bad = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                rejects.append((f, "unreadable")); bad += 1
                continue
            h, w = img.shape[:2]
            det = CoinDetector(params, w, h).detect(img)
            if not det.found:
                rejects.append((f, "no single isolated coin")); bad += 1
        print(f"{label:10s} {len(files):5d} crops  {bad:5d} rejected  {len(files)-bad:5d} clean")

    if args.sheet and rejects:
        tiles = []
        for f, _ in rejects[:64]:
            im = cv2.imread(str(f))
            tiles.append(cv2.resize(im, (128, 128)) if im is not None else np.zeros((128, 128, 3), np.uint8))
        while len(tiles) % 8:
            tiles.append(np.zeros((128, 128, 3), np.uint8))
        cv2.imwrite(args.sheet, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
        print(f"contact sheet of {min(len(rejects), 64)} rejects -> {args.sheet}")

    if args.apply:
        for f, _ in rejects:
            dst = raw / "_rejected" / f.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            f.rename(dst / f.name)
        print(f"moved {len(rejects)} crops under {raw / '_rejected'}")
    elif rejects:
        print("dry run — re-run with --apply to move rejects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
