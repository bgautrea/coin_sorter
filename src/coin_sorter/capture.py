"""Capture training images from the Pi Camera, labelled by class.

Runs on the Raspberry Pi. Writes JPEGs to ``data/raw/<label>/<timestamp>.jpg``.

Typical workflow::

    # Drop a single penny on the belt, run capture in a loop, vary lighting
    python -m coin_sorter.capture --label penny --interval 0.5 --count 200

    # Or with a live preview window
    python -m coin_sorter.capture --label quarter --preview --interval 0.3

The script is deliberately tolerant: if picamera2 is not importable (e.g. you
are iterating on a laptop) it logs a clear error rather than crashing on
import, so the rest of the package stays usable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from . import configure_logging, load_config

log = logging.getLogger("coin_sorter.capture")


def _open_camera(width: int, height: int):  # type: ignore[no-untyped-def]
    """Open and start a Picamera2 still configuration at the given resolution.

    Returns the started camera object. Importing picamera2 is deferred so the
    rest of the package remains importable on non-Pi hosts.
    """
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on host
        log.error(
            "picamera2 is not available. On Bookworm, create your venv with "
            "`python -m venv --system-site-packages .venv` so the system "
            "picamera2 bindings are visible. Underlying error: %s",
            e,
        )
        raise

    picam = Picamera2()
    config = picam.create_still_configuration(main={"size": (width, height)})
    picam.configure(config)
    picam.start()
    # Warm-up — AE/AWB take a moment to converge.
    time.sleep(1.0)
    return picam


def capture_loop(
    label: str,
    interval_s: float,
    count: int | None,
    preview: bool,
    out_root: Path,
    width: int,
    height: int,
) -> int:
    """Capture `count` frames (or forever) at `interval_s` cadence.

    Returns the number of frames actually written.
    """
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing frames to %s", out_dir)

    picam = _open_camera(width, height)
    written = 0
    try:
        if preview:
            try:
                from picamera2.previews import QtPreview  # type: ignore[import-not-found]

                picam.start_preview(QtPreview())
            except Exception as e:  # pragma: no cover - depends on host
                log.warning("Preview requested but unavailable: %s", e)

        while count is None or written < count:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = out_dir / f"{ts}.jpg"
            picam.capture_file(str(path))
            written += 1
            log.info("[%d%s] %s", written, f"/{count}" if count else "", path.name)
            if count is None or written < count:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        try:
            picam.stop()
        except Exception:  # pragma: no cover
            pass
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture labelled training images on the Pi.")
    p.add_argument("--label", required=True, help="Class label / folder name (e.g. penny).")
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between frames.")
    p.add_argument("--count", type=int, default=None, help="Number of frames; omit for endless.")
    p.add_argument("--preview", action="store_true", help="Show a live preview window.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = _parse_args(argv)
    cfg = load_config(args.config)

    width = int(cfg["camera"]["width"])
    height = int(cfg["camera"]["height"])
    raw_dir = Path(cfg["dataset"]["raw_dir"])

    written = capture_loop(
        label=args.label,
        interval_s=args.interval,
        count=args.count,
        preview=args.preview,
        out_root=raw_dir,
        width=width,
        height=height,
    )
    log.info("Done. Wrote %d frames.", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
