"""Main runtime loop for the coin sorter (Pi-side).

Pipeline:
    1. Open the Pi Camera via picamera2.
    2. Open the Pico serial link.
    3. For each frame:
        a. (TODO) Gate on motion / coin presence so we do not classify empty belt.
        b. Run the ONNX classifier.
        c. If confidence > threshold, send ``SORT <label>`` to the Pico.
        d. Cool down for ``sorter.cooldown_ms`` so we do not double-sort.

The stub classifier (when no ONNX model is present) lets us exercise the
camera + serial path end-to-end before training is done.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from . import configure_logging, load_config
from .infer import CoinClassifier
from .pico import Pico, PicoError

log = logging.getLogger("coin_sorter.sorter")


def _open_camera(width: int, height: int):  # type: ignore[no-untyped-def]
    """Open and start a Picamera2 preview-configured stream for inference."""
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        log.error(
            "picamera2 not available. Create the venv with --system-site-packages. (%s)",
            e,
        )
        raise

    picam = Picamera2()
    cfg = picam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(0.5)
    return picam


def _should_classify(_frame: Any) -> bool:
    """Motion / coin-presence gate.

    TODO: implement frame differencing or a lightweight blob detector so we
    only run the classifier when something is actually on the belt. For now
    we run on every frame, which is fine for stub-mode debugging.
    """
    return True


def run(cfg: dict, max_iters: int | None = None) -> None:
    """Run the main loop. `max_iters` is for testing; None = forever."""
    serial_cfg = cfg["serial"]
    sorter_cfg = cfg["sorter"]
    cam_cfg = cfg["camera"]

    classifier = CoinClassifier.from_config(cfg)
    threshold = float(cfg["classifier"]["confidence_threshold"])
    cooldown_s = float(sorter_cfg["cooldown_ms"]) / 1000.0
    target_period_s = 1.0 / float(sorter_cfg["inference_fps"])

    if classifier.is_stub:
        log.warning(
            "Classifier running in STUB mode — no ONNX model loaded. "
            "Train a model and place it at the path in config.yaml to enable real inference."
        )

    picam = _open_camera(int(cam_cfg["width"]), int(cam_cfg["height"]))
    pico = Pico(
        port=serial_cfg["port"],
        baud=int(serial_cfg["baud"]),
        timeout_s=float(serial_cfg["timeout_s"]),
        reconnect_delay_s=float(serial_cfg["reconnect_delay_s"]),
    )

    iters = 0
    try:
        with pico:
            if not pico.ping():
                log.error("Pico did not respond to PING — continuing anyway so you can debug.")
            pico.enable()
            log.info("Sorter loop starting. threshold=%.2f cooldown=%.3fs", threshold, cooldown_s)

            while max_iters is None or iters < max_iters:
                t0 = time.monotonic()
                frame = picam.capture_array()  # HxWx3, RGB888 per config above

                # picamera2 RGB888 actually delivers BGR-ordered bytes for OpenCV
                # compatibility; the classifier expects BGR input.
                if _should_classify(frame):
                    label, conf = classifier.predict(frame)
                    log.info("predict=%s conf=%.3f", label, conf)
                    if conf >= threshold:
                        try:
                            pico.sort(label)
                            time.sleep(cooldown_s)
                        except PicoError as e:
                            log.error("SORT %s failed: %s", label, e)

                # Pace the loop to roughly inference_fps.
                elapsed = time.monotonic() - t0
                if elapsed < target_period_s:
                    time.sleep(target_period_s - elapsed)
                iters += 1
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        try:
            picam.stop()
        except Exception:  # pragma: no cover
            pass
        try:
            pico.disable()
        except Exception:  # pragma: no cover
            pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the coin sorter main loop.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--max-iters", type=int, default=None, help="Stop after N frames (testing).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = _parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, max_iters=args.max_iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
