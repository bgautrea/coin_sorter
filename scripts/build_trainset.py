#!/usr/bin/env python3
"""Project data/raw into a training class set ("view") without moving files.

data/raw is the permanent record: one folder per label, at the finest
granularity you ever classified at. Nothing is ever deleted or renamed there.

A *view* (config: ``dataset.views``) maps those raw folder names onto the
classes you want to train right now. The result is materialised under
``data/processed/<view>/<class>/`` as symlinks, so a view costs no disk and
throwing one away costs nothing either. Rebuild any time; rebuilding replaces
the view directory only, never the raw data.

Why views exist: the Colab notebook trains folder-per-class, so historically
the only way to change the class set was to rename labelled folders - which
loses information. Now the coarse "denomination" model and the fine-grained
model are two views of the same captures.

    python scripts/build_trainset.py --list
    python scripts/build_trainset.py --view denomination
    python scripts/build_trainset.py --view fine_v1 --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_views(cfg_path: Path) -> tuple[dict, Path, Path]:
    cfg = yaml.safe_load(cfg_path.read_text())
    ds = cfg.get("dataset") or {}
    views = ds.get("views") or {}
    if not views:
        raise SystemExit(f"No dataset.views defined in {cfg_path}")
    return views, REPO / ds.get("raw_dir", "data/raw"), REPO / ds.get(
        "processed_dir", "data/processed"
    )


def resolve(raw_name: str, rules: list) -> str | None:
    """First matching rule wins. Returns the target class, or None to exclude.

    A target containing '*' keeps the raw name (so one rule can pass a family
    of labels through unchanged, e.g. quarter_washington_* -> itself).
    """
    for entry in rules:
        pattern, target = entry[0], entry[1]
        if fnmatch.fnmatch(raw_name, pattern):
            if target is None:
                return None
            return raw_name if "*" in target else target
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", help="view name from dataset.views")
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml")
    ap.add_argument("--list", action="store_true", help="list views and exit")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    views, raw_dir, processed_dir = load_views(args.config)

    if args.list or not args.view:
        print("Views in", args.config.name)
        for name, spec in views.items():
            print(f"  {name}  ({len(spec.get('rules') or [])} rules)")
        return 0 if args.list else 2

    if args.view not in views:
        raise SystemExit(f"Unknown view {args.view!r}. Known: {', '.join(views)}")
    rules = views[args.view].get("rules") or []

    if not raw_dir.is_dir():
        raise SystemExit(f"{raw_dir} does not exist")

    # Top-level label folders. Underscore/dot folders are quarantine and are
    # never picked up implicitly - but a rule may name one of their subfolders
    # explicitly as "_triage/cluster", so offer those as candidates too. The
    # catch-all rule at the end of a view excludes anything not named.
    raw_labels = sorted(
        p.name for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )
    for q in sorted(raw_dir.iterdir()):
        if q.is_dir() and q.name.startswith("_") and q.name != "_rejected":
            raw_labels += sorted(f"{q.name}/{c.name}" for c in q.iterdir() if c.is_dir())

    mapping: dict[str, str] = {}
    excluded: list[str] = []
    counts: Counter[str] = Counter()
    empty: list[str] = []

    for label in raw_labels:
        n = sum(1 for p in (raw_dir / label).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        target = resolve(label, rules)
        if target is None:
            excluded.append(f"{label} ({n})")
            continue
        if n == 0:
            empty.append(label)
            continue
        mapping[label] = target
        counts[target] += n

    if not counts:
        raise SystemExit(f"View {args.view!r} selected no images.")

    print(f"View: {args.view}\n")
    width = max(len(c) for c in counts)
    for cls in sorted(counts):
        srcs = sorted(k for k, v in mapping.items() if v == cls)
        detail = "" if srcs == [cls] else "  <- " + ", ".join(srcs)
        print(f"  {cls:<{width}}  {counts[cls]:>5}{detail}")
    print(f"\n  {'TOTAL':<{width}}  {sum(counts.values()):>5} images in {len(counts)} classes")

    if empty:
        print(f"\n  skipped (no images): {', '.join(empty)}")
    if excluded:
        print(f"  excluded by view:    {', '.join(excluded)}")

    smallest = min(counts.values())
    if smallest < 50:
        thin = [c for c in sorted(counts) if counts[c] < 50]
        print(f"\n  WARNING: thin classes (<50): {', '.join(thin)}")

    print("\nclassifier.labels for this view (alphabetical = YOLO class order):")
    for cls in sorted(counts):
        print(f"    - {cls}")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    out = processed_dir / args.view
    if out.exists():
        shutil.rmtree(out)
    for label, target in mapping.items():
        dst = out / target
        dst.mkdir(parents=True, exist_ok=True)
        for src in sorted((raw_dir / label).rglob("*")):
            if src.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # Prefix with the raw label: two source labels folding into one
            # class can contain identically-named files. Flatten any "/" in the
            # label (quarantine sources look like "_triage/cluster") so the
            # prefix stays a filename and does not invent a subdirectory.
            link = dst / f"{label.replace('/', '__')}__{src.name}"
            if not link.exists():
                link.symlink_to(src.resolve())

    print(f"\nWrote {out} ({sum(counts.values())} symlinks). data/raw untouched.")
    print(f"Bundle it with:  ./scripts/zip_dataset.sh --view {args.view}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
