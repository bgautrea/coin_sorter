"""Main runtime loop for the coin sorter (Pi-side).

Pipeline:
    1. Open the Pi Camera via picamera2.
    2. Open the Pico serial link.
    3. For each frame:
        a. Crop to the ROI and run ``CoinDetector`` — the same presence gate the
           capture path uses, so empty belt costs one blob detection and no
           inference at all.
        b. Ask the ``Deduper`` whether this frame is the one to act on. In
           ``centroid_band`` mode a coin fires exactly once, when its centroid
           crosses the trip line, re-arming only when the belt goes empty.
        c. Crop tightly around the coin with ``tight_square_crop`` — the model
           is trained on tight crops, so handing it a full frame would be a
           train/serve mismatch.
        d. Run the ONNX classifier on that crop.
        e. Map the label to a physical bin via ``sorter.sort_map`` (unknown
           labels and low confidence -> ``check``) and send ``SORT <bin>``.
        f. Cool down for ``sorter.cooldown_ms`` so we do not double-sort.

The stub classifier (when no ONNX model is present) lets us exercise the
camera + serial path end-to-end before training is done.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import configure_logging, load_config
from .infer import CoinClassifier
from .pico import Pico, PicoError

log = logging.getLogger("coin_sorter.sorter")


def _open_camera(
    width: int,
    height: int,
    af_mode: str = "continuous",
    lens_position=None,
    awb_mode: str = "auto",
    colour_gains=None,
    exposure_us=None,
    analogue_gain=None,
):  # type: ignore[no-untyped-def]
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
    from .capture import apply_autofocus, apply_exposure, apply_white_balance

    apply_autofocus(picam, af_mode, lens_position)
    apply_white_balance(picam, awb_mode, colour_gains)
    apply_exposure(picam, exposure_us, analogue_gain)
    time.sleep(0.5)
    return picam


def run(
    cfg: dict,
    max_iters: int | None = None,
    belt_hz: int | None = None,
    force_bin: str | None = None,
    no_sort: bool = False,
    save_crops: Path | None = None,
) -> None:
    """Run the main loop.

    `max_iters` stops after N frames. The rest are bench-test aids:
        belt_hz     free-run the belt at this rate (None = leave it alone)
        force_bin   ignore the classifier and always sort here, for timing runs
        no_sort     gate, crop and classify but never command the diverter
        save_crops  write every crop the classifier saw, named with its verdict
    """
    serial_cfg = cfg["serial"]
    sorter_cfg = cfg["sorter"]
    cam_cfg = cfg["camera"]

    classifier = CoinClassifier.from_config(cfg)
    threshold = float(cfg["classifier"]["confidence_threshold"])
    label_to_bin = {
        lab: b for b, labs in (sorter_cfg.get("sort_map") or {}).items() for lab in (labs or [])
    }
    cooldown_s = float(sorter_cfg["cooldown_ms"]) / 1000.0
    target_period_s = 1.0 / float(sorter_cfg["inference_fps"])

    # Presence gate + tight crop, reusing the capture path so the frames the
    # model sees at run time are framed exactly like the ones it trained on.
    from .capture import CoinDetector, Deduper, crop_roi, resolve_roi, tight_square_crop

    cap_params = cfg.get("capture", {})
    roi_px = resolve_roi(cap_params.get("roi") or cam_cfg.get("roi"),
                         int(cam_cfg["width"]), int(cam_cfg["height"]))
    roi_w = roi_px[2] if roi_px else int(cam_cfg["width"])
    roi_h = roi_px[3] if roi_px else int(cam_cfg["height"])
    detector = CoinDetector(cap_params, roi_w, roi_h)
    # One decision per physical coin, regardless of what bulk capture was set to.
    deduper = Deduper("centroid_band", cap_params, roi_w, roi_h)
    pad_frac = float(cap_params.get("crop_pad_frac", 0.25))
    crop_square = bool(cap_params.get("crop_square", True))
    gated = classified = 0
    latencies: list[float] = []
    if save_crops:
        save_crops.mkdir(parents=True, exist_ok=True)

    if classifier.is_stub:
        log.warning(
            "Classifier running in STUB mode — no ONNX model loaded. "
            "Train a model and place it at the path in config.yaml to enable real inference."
        )

    picam = _open_camera(
        int(cam_cfg["width"]),
        int(cam_cfg["height"]),
        cam_cfg.get("af_mode", "continuous"),
        cam_cfg.get("lens_position"),
        cam_cfg.get("awb", "auto"),
        cam_cfg.get("colour_gains"),
        cam_cfg.get("exposure_us"),
        cam_cfg.get("analogue_gain"),
    )
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
            if belt_hz:
                # The trip-line gate needs coins in motion; SORT no longer
                # halts the free-run (firmware SORT_ADVANCES_BELT = False).
                pico.run(int(belt_hz))
                log.info("Belt free-running at %d Hz", belt_hz)
            log.info(
                "Sorter loop starting. threshold=%.2f cooldown=%.3fs%s%s",
                threshold, cooldown_s,
                f" force_bin={force_bin}" if force_bin else "",
                " NO-SORT" if no_sort else "",
            )

            while max_iters is None or iters < max_iters:
                t0 = time.monotonic()
                frame = picam.capture_array()  # HxWx3, RGB888 per config above

                # picamera2 RGB888 actually delivers BGR-ordered bytes for OpenCV
                # compatibility; the detector and classifier both expect BGR.
                roi = crop_roi(frame, roi_px)
                det = detector.detect(roi)
                if det.found:
                    gated += 1

                if deduper.should_save(det, time.monotonic()):
                    crop = tight_square_crop(roi, det, pad_frac, crop_square)
                    if crop is None:
                        log.debug("coin at %s produced an empty crop; skipped", det.centroid)
                    else:
                        classified += 1
                        t_trip = time.monotonic()
                        label, conf = classifier.predict(crop)
                        bin_ = label_to_bin.get(label, "check") if conf >= threshold else "check"
                        if force_bin:
                            bin_ = force_bin
                        t_infer = time.monotonic() - t_trip
                        if save_crops:
                            import cv2

                            name = f"{int(t_trip * 1000)}_{label}_{conf:.2f}_{bin_}.jpg"
                            cv2.imwrite(str(save_crops / name), crop)
                        if no_sort:
                            log.info(
                                "predict=%s conf=%.3f -> %s (NOT sent) "
                                "infer=%.0fms centroid=%s fill=%.2f",
                                label, conf, bin_, t_infer * 1000, det.centroid, det.fill,
                            )
                        else:
                            try:
                                pico.sort(bin_)
                                total = time.monotonic() - t_trip
                                latencies.append(total)
                                log.info(
                                    "predict=%s conf=%.3f -> %s  "
                                    "infer=%.0fms trip->aimed=%.0fms centroid=%s fill=%.2f",
                                    label, conf, bin_, t_infer * 1000, total * 1000,
                                    det.centroid, det.fill,
                                )
                                time.sleep(cooldown_s)
                            except PicoError as e:
                                log.error("SORT %s (%s) failed: %s", bin_, label, e)

                # Pace the loop to roughly inference_fps.
                elapsed = time.monotonic() - t0
                if elapsed < target_period_s:
                    time.sleep(target_period_s - elapsed)
                iters += 1
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        # Gate ratio is the health check: on a running belt `classified` should
        # track the number of coins that went past, not the frame count.
        log.info(
            "%d frames, %d with a coin present, %d classified/sorted.",
            iters, gated, classified,
        )
        if latencies:
            latencies.sort()
            log.info(
                "trip->aimed latency: min %.0f ms, median %.0f ms, max %.0f ms. "
                "The dish must reach its bin before the coin reaches the nose; "
                "compare against (trip-line -> nose distance) / belt speed.",
                latencies[0] * 1000,
                latencies[len(latencies) // 2] * 1000,
                latencies[-1] * 1000,
            )
        if belt_hz:
            try:
                pico.stop()
            except Exception:  # pragma: no cover
                pass
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
    p.add_argument("--belt-hz", type=int, default=None,
                   help="Free-run the belt at this rate. Omit to leave it alone.")
    p.add_argument("--force-bin", choices=("keep", "common", "check"), default=None,
                   help="Ignore the classifier and always sort here (timing runs).")
    p.add_argument("--no-sort", action="store_true",
                   help="Gate, crop and classify, but never command the diverter.")
    p.add_argument("--save-crops", type=Path, default=None,
                   help="Write every crop the classifier saw, named with its verdict.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = _parse_args(argv)
    cfg = load_config(args.config)
    run(
        cfg,
        max_iters=args.max_iters,
        belt_hz=args.belt_hz,
        force_bin=args.force_bin,
        no_sort=args.no_sort,
        save_crops=args.save_crops,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
