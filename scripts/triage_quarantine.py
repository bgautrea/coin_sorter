#!/usr/bin/env python3
"""Sort data/raw/_rejected/ into buckets by WHY each crop failed the gate.

prune_crops.py quarantines everything under one generic reason ("no single
isolated coin"), but the contents are not homogeneous: alongside genuine
clusters and half-coins there are plenty of clean, well-centred coins that
merely tripped a tight border or isolation test. Training a `reject` class on
the lot would teach the model that a clean penny is junk.

So re-run the same segmentation and record which gate actually failed:

    cluster   another blob shares the crop -> two+ coins. A real reject.
    partial   the coin touches the crop border -> cut off. A real reject.
    overlap   fill fraction too low -> coins on top of each other. A real reject.
    blank     nothing coin-like at all (dark/blurred frame). A real reject.
    shape     failed circularity or area only - usually motion blur.
    clean     passes every gate - was quarantined by a stricter threshold and
              is a perfectly good training image for its original label.

Nothing is moved or deleted. Buckets are SYMLINKS under data/raw/_triage/, so
the quarantine stays exactly as it is and a bucket costs nothing to rebuild.

    python scripts/triage_quarantine.py                # report only
    python scripts/triage_quarantine.py --apply        # write _triage/ links
    python scripts/triage_quarantine.py --sheets out/  # contact sheet per bucket
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from coin_sorter import load_config  # noqa: E402
from coin_sorter.capture import CoinDetector  # noqa: E402

# Same bracket prune_crops.py used, so triage agrees with what quarantined them.
CROP_PARAMS = {
    "min_area_frac": 0.15, "max_area_frac": 0.75,
    "min_fill_frac": 0.85, "reject_border": True, "isolate": True,
    "crop_pad_frac": 0.0,
    "isolate_min_frac": 0.001,
}
# Buckets that are genuinely what a `reject` class should contain.
REJECT_BUCKETS = ("cluster", "partial", "overlap", "blank")


def classify(img, params) -> str:
    """Re-run the gates individually and name the first one that fails."""
    h, w = img.shape[:2]
    det = CoinDetector(params, w, h)
    if det.detect(img).found:
        return "clean"

    # Reproduce _pick_blob's stages so we can see which test rejected it.
    import math

    gray = det._to_gray(img)
    mask = det._mask_absdiff(gray) if det.method == "absdiff" else det._mask_brightness(gray)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, det._kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, det._kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [(cv2.contourArea(c), c, cv2.boundingRect(c)) for c in cnts]
    blobs = [b for b in blobs if b[0] >= det.isolate_min_area]
    if not blobs:
        return "blank"

    sized = [b for b in blobs if det.min_area <= b[0] <= det.max_area]
    if not sized:
        return "shape"

    # More than one coin-sized blob is a cluster regardless of anything else.
    if len(sized) > 1:
        return "cluster"

    area, c, (bx, by, bw, bh) = sized[0]
    rh, rw = mask.shape[:2]
    perim = cv2.arcLength(c, True)
    circ = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    (_, _), r = cv2.minEnclosingCircle(c)
    fill = area / (math.pi * r * r) if r > 0 else 0.0

    if bx <= 0 or by <= 0 or bx + bw >= rw or by + bh >= rh:
        return "partial"
    if fill < det.min_fill:
        return "overlap"
    if circ < det.min_circularity:
        return "shape"
    if not det._isolated((bx, by, bw, bh), c, blobs):
        return "cluster"
    return "shape"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--apply", action="store_true", help="write data/raw/_triage/ symlinks")
    ap.add_argument("--sheets", type=Path, help="write one contact sheet per bucket here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    params = {**cfg.get("capture", {}), **CROP_PARAMS}
    raw = REPO / cfg["dataset"]["raw_dir"]
    quarantine = raw / "_rejected"
    if not quarantine.is_dir():
        raise SystemExit(f"{quarantine} does not exist")

    buckets: dict[str, list[Path]] = defaultdict(list)
    per_source: dict[str, Counter] = defaultdict(Counter)

    for sub in sorted(p for p in quarantine.iterdir() if p.is_dir()):
        for f in sorted(sub.glob("*.jpg")):
            img = cv2.imread(str(f))
            bucket = "blank" if img is None else classify(img, params)
            buckets[bucket].append(f)
            per_source[sub.name][bucket] += 1

    order = ["cluster", "partial", "overlap", "blank", "shape", "clean"]
    total = sum(len(v) for v in buckets.values())
    print(f"{total} quarantined crops in {quarantine}\n")
    print(f"  {'bucket':<9} {'count':>6}   {'':<4}")
    for b in order:
        if buckets.get(b):
            tag = "-> reject" if b in REJECT_BUCKETS else ""
            print(f"  {b:<9} {len(buckets[b]):>6}   {tag}")
    usable = sum(len(buckets.get(b, [])) for b in REJECT_BUCKETS)
    print(f"\n  usable as `reject`: {usable}")
    print(f"  recoverable (clean): {len(buckets.get('clean', []))}")

    print("\n  by source folder:")
    for src, c in sorted(per_source.items()):
        detail = "  ".join(f"{b}={c[b]}" for b in order if c[b])
        print(f"    {src:<16} {detail}")

    if args.sheets:
        from PIL import Image

        args.sheets.mkdir(parents=True, exist_ok=True)
        for b, files in buckets.items():
            pick, cell = files[:24], 150
            sheet = Image.new("RGB", (cell * 6, cell * 4), "black")
            for i, f in enumerate(pick):
                im = Image.open(f).convert("RGB")
                im.thumbnail((cell, cell))
                sheet.paste(im, ((i % 6) * cell, (i // 6) * cell))
            sheet.save(args.sheets / f"{b}.jpg", quality=88)
        print(f"\n  contact sheets -> {args.sheets}")

    if not args.apply:
        print("\n(report only — re-run with --apply to write data/raw/_triage/)")
        return 0

    triage = raw / "_triage"
    if triage.exists():
        shutil.rmtree(triage)
    for b, files in buckets.items():
        d = triage / b
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            link = d / f"{f.parent.name}__{f.name}"
            if not link.exists():
                link.symlink_to(f.resolve())
    print(f"\nWrote {triage} ({total} symlinks). _rejected/ untouched.")
    print("Add to a view with a rule like:  - [\"_triage/cluster\", \"reject\"]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
