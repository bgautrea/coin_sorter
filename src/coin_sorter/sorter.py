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
           labels and low confidence -> ``check``) and SCHEDULE the divert for
           when the coin actually reaches the nose, ``sorter.transport_delay_s``
           later. The camera sits far upstream of the diverter -- ~240 mm, some
           30 s of belt at 550 Hz -- so a dozen or more coins are in flight at
           once. Aiming the dish on detection would put every coin in whichever
           bin the newest coin upstream asked for.
        f. Drain due diverts from that queue each iteration.

The stub classifier (when no ONNX model is present) lets us exercise the
camera + serial path end-to-end before training is done.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
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


class DivertQueue:
    """Coins in flight between the camera's trip line and the roller nose.

    The classifier decides ~240 mm upstream of the diverter, which at ~7.9 mm/s
    is about 30 s of belt -- so a dozen or more coins are travelling at any
    moment. Aiming the dish when a coin is *classified* would put every coin in
    whichever bin the newest coin upstream asked for. Instead each decision is
    scheduled for the coin's own arrival.

    A coin whose window has passed by more than ``hold_s`` is reported missed
    rather than diverted late: by then it has already tipped off the nose, and
    aiming the dish for it would only mis-sort whichever coin is there now.
    """

    def __init__(self, delay_s: float, hold_s: float, lead_s: float = 0.0) -> None:
        self.delay_s = float(delay_s)
        self.hold_s = float(hold_s)
        # Aim early: the dish has to be SETTLED when the coin tips off, not
        # still slewing. The lead covers servo travel plus belt-speed drift.
        self.lead_s = float(lead_s)
        self._q: deque[tuple[float, str, str]] = deque()

    def __len__(self) -> int:
        return len(self._q)

    def schedule(self, bin_: str, label: str, now: float) -> float:
        """Queue a divert for a coin detected at `now`. Returns its due time."""
        due = now + self.delay_s - self.lead_s
        self._q.append((due, bin_, label))
        return due

    def next_due(self) -> float | None:
        """When the next divert comes due, or None if nothing is in flight."""
        return self._q[0][0] if self._q else None

    def due(self, now: float) -> list[tuple[str, str, float, bool]]:
        """Pop everything whose time has come.

        Returns (bin, label, lateness, missed) per coin, oldest first. FIFO is
        correct here without sorting: the belt cannot reorder coins, so
        detection order is arrival order.
        """
        out = []
        while self._q and self._q[0][0] <= now:
            due_at, bin_, label = self._q.popleft()
            late = now - due_at
            out.append((bin_, label, late, late > self.hold_s))
        return out


def run(
    cfg: dict,
    max_iters: int | None = None,
    belt_hz: int | None = None,
    force_bin: str | None = None,
    no_sort: bool = False,
    save_crops: Path | None = None,
    transport_delay_s: float | None = None,
    measure_transport: bool = False,
) -> None:
    """Run the main loop.

    `max_iters` stops after N frames. The rest are bench-test aids:
        belt_hz     free-run the belt at this rate (None = leave it alone)
        force_bin   ignore the classifier and always sort here, for timing runs
        no_sort     gate, crop and classify but never command the diverter
        save_crops  write every crop the classifier saw, named with its verdict
        transport_delay_s  override sorter.transport_delay_s
        measure_transport  log detections with a wall clock and queue nothing,
                    so you can time a single coin from trip line to nose
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
    if transport_delay_s is None:
        transport_delay_s = float(sorter_cfg.get("transport_delay_s", 0.0))
    hold_s = float(sorter_cfg.get("divert_hold_s", 0.5))
    lead_s = float(sorter_cfg.get("divert_lead_s", 0.0))
    neutral_bin = sorter_cfg.get("neutral_bin") or None
    neutral_after_s = float(sorter_cfg.get("neutral_after_s", 1.0))
    queue = DivertQueue(transport_delay_s, hold_s, lead_s)
    dish_bin: str | None = None      # where the dish is currently aimed
    last_divert_at: float | None = None

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
    gated = classified = missed = 0
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
                        if measure_transport:
                            log.info(
                                "TRIPPED at %s  predict=%s conf=%.3f -> %s. "
                                "Time it to the nose; that is transport_delay_s.",
                                time.strftime("%H:%M:%S"), label, conf, bin_,
                            )
                        elif no_sort:
                            log.info(
                                "predict=%s conf=%.3f -> %s (NOT sent, due in %.1fs) "
                                "infer=%.0fms centroid=%s fill=%.2f",
                                label, conf, bin_, transport_delay_s,
                                t_infer * 1000, det.centroid, det.fill,
                            )
                        else:
                            # Schedule, do not aim: the coin is ~transport_delay_s
                            # away from the nose and other coins are ahead of it.
                            queue.schedule(bin_, label, t_trip)
                            log.info(
                                "predict=%s conf=%.3f -> %s queued (due in %.1fs, "
                                "%d in flight) infer=%.0fms fill=%.2f",
                                label, conf, bin_, transport_delay_s, len(queue),
                                t_infer * 1000, det.fill,
                            )
                        time.sleep(cooldown_s)

                # Drain any coin that has now reached the nose. Checked every
                # iteration, not only on detection, because the belt keeps
                # delivering long after the last coin was classified.
                for bin_, label, late, was_missed in queue.due(time.monotonic()):
                    if was_missed:
                        log.warning(
                            "MISSED %s (%s): due %.1fs ago, dish not aimed in time",
                            label, bin_, late,
                        )
                        missed += 1
                        continue
                    try:
                        pico.sort(bin_)
                        dish_bin = bin_
                        last_divert_at = time.monotonic()
                        latencies.append(transport_delay_s - lead_s + late)
                        log.info(
                            "divert -> %s for %s (%.2fs late, %d still in flight)",
                            bin_, label, late, len(queue),
                        )
                        time.sleep(hold_s)
                    except PicoError as e:
                        log.error("SORT %s (%s) failed: %s", bin_, label, e)

                # Park at neutral once the coin has left the dish, so anything
                # arriving unscheduled falls to the default bin instead of the
                # last sorted coin's. Skipped when another divert is imminent,
                # which would only bounce the servo there and back.
                if neutral_bin and dish_bin not in (None, neutral_bin) and last_divert_at:
                    now = time.monotonic()
                    nxt = queue.next_due()
                    if now - last_divert_at >= neutral_after_s and (
                        nxt is None or nxt - now > neutral_after_s
                    ):
                        try:
                            pico.sort(neutral_bin)
                            log.debug("dish returned to neutral (%s)", neutral_bin)
                            dish_bin = neutral_bin
                            last_divert_at = None
                        except PicoError as e:
                            log.error("SORT %s (neutral) failed: %s", neutral_bin, e)

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
                "scheduled->diverted: min %.1fs median %.1fs max %.1fs "
                "(target %.1fs = transport %.1fs - lead %.1fs)",
                latencies[0], latencies[len(latencies) // 2], latencies[-1],
                transport_delay_s - lead_s, transport_delay_s, lead_s,
            )
        if len(queue):
            log.warning(
                "%d coin(s) still in flight at shutdown -- they will land in "
                "whatever bin the dish was last set to. Let the belt run "
                "transport_delay_s (%.0fs) longer than the last coin.",
                len(queue), transport_delay_s,
            )
        if missed:
            log.warning(
                "%d coin(s) missed their divert window (>%.1fs late). If this is "
                "not zero, the loop is stalling -- lower inference_fps work or "
                "raise divert_hold_s.", missed, hold_s,
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
    p.add_argument("--transport-delay", type=float, default=None,
                   help="Seconds from the trip line to the nose. Overrides "
                        "sorter.transport_delay_s. Measure it, do not compute it.")
    p.add_argument("--measure-transport", action="store_true",
                   help="Log each detection with a wall clock and queue nothing, "
                        "so you can time one coin from trip line to nose.")
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
        transport_delay_s=args.transport_delay,
        measure_transport=args.measure_transport,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
