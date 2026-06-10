"""Capture training images from the Pi Camera, labelled by class.

Runs on the Raspberry Pi. Writes JPEGs to ``data/raw/<label>/<timestamp>.jpg``.

The rig is belt-fed: a single denomination is loaded and the belt runs while
coins pass under the camera. A coin-presence gate (see :class:`CoinDetector`)
ensures we only save frames that actually contain a coin, tight-cropped so the
coin fills the saved image — which is what the downstream YOLO-cls whole-image
classifier wants once it squashes everything to 224x224.

Typical workflow::

    # One-time: find the belt ROI and tune thresholds (writes debug snapshots).
    python -m coin_sorter.capture --label _calib --calibrate

    # Capture ~250 coins of one denomination (load the hopper, run the belt):
    python -m coin_sorter.capture --label penny --count 250

    # The 'reject' class bypasses the single-coin gate automatically — feed
    # debris, multi-coin clusters and empty belt:
    python -m coin_sorter.capture --label reject --interval 0.4 --count 250

Note: with the gate on (the default), ``--count`` counts *coins saved*, not
frames grabbed. Pass ``--no-gate`` to fall back to the legacy "save the
ROI-cropped frame every --interval" behaviour.

The script is deliberately tolerant: if picamera2 is not importable (e.g. you
are iterating on a laptop) it logs a clear error rather than crashing on
import, so the rest of the package stays usable.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import configure_logging, load_config

log = logging.getLogger("coin_sorter.capture")


def apply_autofocus(picam, af_mode: str = "continuous", lens_position=None) -> None:  # type: ignore[no-untyped-def]
    """Set the Arducam 64MP autofocus via libcamera controls (picamera2).

    af_mode: ``manual`` locks the lens at ``lens_position`` dioptres (0=infinity
    .. ~15=close — same scale as cam_test/stream.py); ``auto`` does a one-shot
    focus then locks; ``continuous`` keeps refocusing. Failures are logged and
    swallowed so capture still runs at the default lens position.
    """
    try:
        from libcamera import controls  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on host
        log.warning("libcamera controls unavailable (%s); leaving focus at default.", e)
        return
    mode = (af_mode or "continuous").lower()
    try:
        if mode == "manual":
            lp = float(lens_position or 0.0)
            picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lp})
            log.info("Focus: manual, lens_position=%.2f dioptres", lp)
        elif mode == "auto":
            picam.set_controls({"AfMode": controls.AfModeEnum.Auto})
            picam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
            time.sleep(2.0)  # let the one-shot sweep converge
            log.info("Focus: one-shot autofocus complete.")
        else:
            picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            log.info("Focus: continuous autofocus.")
    except Exception as e:  # pragma: no cover - depends on hardware
        log.warning("Could not set autofocus (%s); leaving focus at default.", e)


def apply_white_balance(picam, awb_mode: str = "auto", colour_gains=None) -> None:  # type: ignore[no-untyped-def]
    """Set white balance via libcamera controls (picamera2).

    ``manual`` locks AWB and applies ``colour_gains`` as (red, blue) multipliers
    — needed for consistent colour under the WS2812 ring, whose narrow spectrum
    fools auto-WB into a colour cast. ``auto`` re-enables AWB. Failures are
    logged and swallowed so capture still runs.
    """
    try:
        from libcamera import controls  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on host
        log.warning("libcamera controls unavailable (%s); leaving white balance at default.", e)
        return
    try:
        if (awb_mode or "auto").lower() == "manual" and colour_gains:
            r, b = (float(x) for x in colour_gains)
            picam.set_controls({"AwbEnable": False, "ColourGains": (r, b)})
            log.info("White balance: manual, gains red=%.2f blue=%.2f", r, b)
        else:
            picam.set_controls({"AwbEnable": True})
            log.info("White balance: auto.")
    except Exception as e:  # pragma: no cover - depends on hardware
        log.warning("Could not set white balance (%s); leaving at default.", e)


def apply_exposure(picam, exposure_us=None, analogue_gain=None) -> None:  # type: ignore[no-untyped-def]
    """Set exposure time (microseconds) and analogue gain (picamera2 controls).

    Both ``None`` ⇒ auto-exposure. Otherwise auto-exposure is locked and a fixed
    ``ExposureTime`` / ``AnalogueGain`` applied — a short exposure freezes the
    motion blur a moving belt causes (the bright ring light makes a fast shutter
    feasible). Failures are logged and swallowed.
    """
    try:
        if exposure_us is None and analogue_gain is None:
            picam.set_controls({"AeEnable": True})
            log.info("Exposure: auto.")
        else:
            c = {"AeEnable": False}
            if exposure_us is not None:
                c["ExposureTime"] = int(exposure_us)
            if analogue_gain is not None:
                c["AnalogueGain"] = float(analogue_gain)
            picam.set_controls(c)
            log.info("Exposure: manual, time=%sus gain=%s", c.get("ExposureTime"), c.get("AnalogueGain"))
    except Exception as e:  # pragma: no cover - depends on hardware
        log.warning("Could not set exposure (%s); leaving at default.", e)


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
    """Open and start a Picamera2 preview-configured RGB888 stream.

    We use the preview configuration (not still) because the belt-fed loop
    grabs frames continuously rather than taking one high-res shot. RGB888 on
    picamera2 delivers BGR-ordered bytes, so ``capture_array`` yields exactly
    what OpenCV (and ``cv2.imwrite``) expect — and it matches the colour path
    in ``sorter.py``. Importing picamera2 is deferred so the rest of the
    package remains importable on non-Pi hosts.
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
    config = picam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
    )
    picam.configure(config)
    picam.start()
    apply_autofocus(picam, af_mode, lens_position)
    apply_white_balance(picam, awb_mode, colour_gains)
    apply_exposure(picam, exposure_us, analogue_gain)
    # Warm-up — AE/AWB take a moment to converge.
    time.sleep(1.0)
    return picam


# --------------------------------------------------------------------------- #
# ROI helpers
# --------------------------------------------------------------------------- #


def resolve_roi(
    roi_cfg: Any, frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    """Convert a fractional ``[x, y, w, h]`` ROI (0..1) to clamped pixels.

    Returns ``None`` (meaning "use the whole frame") if ``roi_cfg`` is falsy.
    """
    if not roi_cfg:
        return None
    x_f, y_f, w_f, h_f = (float(v) for v in roi_cfg)
    x = int(round(x_f * frame_w))
    y = int(round(y_f * frame_h))
    w = int(round(w_f * frame_w))
    h = int(round(h_f * frame_h))
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h)


def crop_roi(frame: np.ndarray, roi_px: tuple[int, int, int, int] | None) -> np.ndarray:
    """Return the ROI sub-image, or the whole frame if ``roi_px`` is None."""
    if roi_px is None:
        return frame
    x, y, w, h = roi_px
    return frame[y : y + h, x : x + w]


# --------------------------------------------------------------------------- #
# Coin presence detection
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """Result of a single-frame coin-presence check, in ROI pixel coords."""

    found: bool
    centroid: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    area: float = 0.0
    circularity: float = 0.0
    mask: np.ndarray | None = field(default=None, repr=False)


class CoinDetector:
    """Segment a single round coin against a moving, roughly-uniform belt.

    The belt *surface moves*, so frame-differencing against a static reference
    fires everywhere. The primary ``brightness`` method judges each frame
    independently (Otsu / fixed intensity threshold), which is motion-
    invariant; a contour area-fraction + circularity filter is what actually
    decides "coin present" and rejects belt seams, shadows and debris streaks.
    The ``absdiff`` method (rolling-median background) is a fallback for when
    the coin and belt are too close in brightness to threshold.
    """

    def __init__(self, params: dict, roi_w: int, roi_h: int) -> None:
        self.method = str(params.get("method", "brightness"))
        k = int(params.get("blur_ksize", 5))
        if k % 2 == 0:
            k += 1
        self.blur_ksize = max(1, k)

        roi_area = float(roi_w * roi_h)
        self.min_area = float(params.get("min_area_frac", 0.02)) * roi_area
        self.max_area = float(params.get("max_area_frac", 0.60)) * roi_area
        self.min_circularity = float(params.get("min_circularity", 0.65))

        bt = params.get("bright_thresh", None)
        self.bright_thresh = None if bt is None else float(bt)
        self.invert = bool(params.get("invert", False))

        self.bg_window = max(3, int(params.get("bg_window", 25)))
        self.absdiff_thresh = int(params.get("absdiff_thresh", 25))
        self._bg: deque[np.ndarray] = deque(maxlen=self.bg_window)

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _to_gray(self, roi_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        return gray

    def _mask_brightness(self, gray: np.ndarray) -> np.ndarray:
        thr_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        if self.bright_thresh is None:
            _, mask = cv2.threshold(gray, 0, 255, thr_type | cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, self.bright_thresh, 255, thr_type)
        return mask

    def _mask_absdiff(self, gray: np.ndarray) -> np.ndarray:
        self._bg.append(gray)
        # Wait until the rolling window is reasonably full so the median is
        # dominated by belt, not by the first coin we happen to see.
        if len(self._bg) < max(3, self.bg_window // 2):
            return np.zeros_like(gray)
        bg = np.median(np.stack(self._bg, axis=0), axis=0).astype(np.uint8)
        diff = cv2.absdiff(gray, bg)
        _, mask = cv2.threshold(diff, self.absdiff_thresh, 255, cv2.THRESH_BINARY)
        return mask

    def detect(self, roi_bgr: np.ndarray) -> Detection:
        """Return a :class:`Detection` for the largest coin-like blob (if any)."""
        gray = self._to_gray(roi_bgr)
        mask = self._mask_absdiff(gray) if self.method == "absdiff" else self._mask_brightness(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return self._pick_blob(mask)

    def _pick_blob(self, mask: np.ndarray) -> Detection:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = 0.0
        best: Detection | None = None
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < self.min_circularity:
                continue
            if area <= best_area:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = int(round(m["m10"] / m["m00"]))
            cy = int(round(m["m01"] / m["m00"]))
            best_area = area
            best = Detection(
                found=True,
                centroid=(cx, cy),
                bbox=tuple(cv2.boundingRect(c)),  # type: ignore[arg-type]
                area=area,
                circularity=circ,
                mask=mask,
            )
        if best is None:
            return Detection(found=False, mask=mask)
        return best


# --------------------------------------------------------------------------- #
# Dedup — one physical coin spans many consecutive frames
# --------------------------------------------------------------------------- #


class Deduper:
    """Decide which of many coin-present frames to actually save.

    Modes:
        centroid_band — save once, when the coin centroid first crosses a
                        virtual trip line; re-arm only when the ROI goes empty.
        min_interval  — save at most once per ``min_interval_s`` while present.
        none          — save every gated frame (many near-duplicates).
    """

    def __init__(self, mode: str, params: dict, roi_w: int, roi_h: int) -> None:
        self.mode = mode
        self.travel_axis = str(params.get("travel_axis", "y"))
        self.min_interval_s = float(params.get("min_interval_s", 0.6))

        axis_len = roi_h if self.travel_axis == "y" else roi_w
        center = float(params.get("band_center_frac", 0.5)) * axis_len
        half = float(params.get("band_halfwidth_frac", 0.08)) * axis_len
        self.band_lo = center - half
        self.band_hi = center + half

        self._armed = True
        self._last_save_s: float | None = None

    def _coord(self, det: Detection) -> int:
        cx, cy = det.centroid  # type: ignore[misc]
        return cy if self.travel_axis == "y" else cx

    def should_save(self, det: Detection, now_s: float) -> bool:
        if self.mode == "none":
            return det.found

        if not det.found:
            # Empty ROI re-arms the band trigger so the next coin can fire.
            if self.mode == "centroid_band":
                self._armed = True
            return False

        if self.mode == "centroid_band":
            if not self._armed:
                return False
            coord = self._coord(det)
            if not (self.band_lo <= coord <= self.band_hi):
                return False

        # Rate cap: per-coin cadence in min_interval mode, jam safety-valve in
        # centroid_band mode (a coin stuck in the band stays disarmed anyway).
        if self._last_save_s is not None and (now_s - self._last_save_s) < self.min_interval_s:
            return False

        self._last_save_s = now_s
        if self.mode == "centroid_band":
            self._armed = False
        return True


# --------------------------------------------------------------------------- #
# Tight crop
# --------------------------------------------------------------------------- #


def tight_square_crop(
    roi_bgr: np.ndarray, det: Detection, pad_frac: float, square: bool
) -> np.ndarray | None:
    """Crop tightly around the detected coin so it fills the saved image.

    Pads the bbox by ``pad_frac`` of its longest side, optionally makes the
    crop square around the *centroid* (stable under partial occlusion), then
    shifts the window fully inside the ROI before clamping. Returns ``None`` if
    the resulting crop would be empty.
    """
    if det.bbox is None or det.centroid is None:
        return None
    h, w = roi_bgr.shape[:2]
    bx, by, bw, bh = det.bbox
    pad = int(round(pad_frac * max(bw, bh)))
    x0, y0 = bx - pad, by - pad
    x1, y1 = bx + bw + pad, by + bh + pad

    if square:
        side = max(x1 - x0, y1 - y0)
        cx, cy = det.centroid
        x0, y0 = cx - side // 2, cy - side // 2
        x1, y1 = x0 + side, y0 + side

    # Shift fully inside the ROI where possible (preserves squareness)...
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    # ...then clamp (coin larger than ROI on an axis just yields a rectangle;
    # training squashes to 224x224 so mild non-squareness is fine).
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = roi_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop.copy()


# --------------------------------------------------------------------------- #
# Pico: ring light + optional belt driving
# --------------------------------------------------------------------------- #


def maybe_open_pico(
    serial_cfg: dict,
    *,
    lights: bool,
    light_rgb,
    drive_belt: bool,
    belt_speed_hz: int,
):  # type: ignore[no-untyped-def]
    """Best-effort: open the Pico for the ring light and/or belt driving.

    Turns the WS2812 ring on (for even, consistent illumination) and, if
    ``drive_belt``, starts the belt free-running via the firmware's non-blocking
    ``RUN``. Returns the open :class:`~coin_sorter.pico.Pico` or ``None``; any
    failure is logged and swallowed so capture proceeds without lights/belt.
    """
    if not (lights or drive_belt):
        return None
    try:
        from .pico import Pico

        pico = Pico(
            port=serial_cfg["port"],
            baud=int(serial_cfg["baud"]),
            timeout_s=float(serial_cfg["timeout_s"]),
            reconnect_delay_s=float(serial_cfg["reconnect_delay_s"]),
        )
        pico.open()
        if not pico.ping():
            log.warning("Pico did not respond to PING — no ring light / belt.")
            pico.close()
            return None
        if lights:
            r, g, b = (int(v) for v in light_rgb)
            pico.set_leds(r, g, b)
            log.info("Ring light on: rgb=(%d, %d, %d).", r, g, b)
        if drive_belt:
            pico.run(int(belt_speed_hz))
            log.info("Belt running continuously at %d Hz.", belt_speed_hz)
        return pico
    except Exception as e:  # pragma: no cover - depends on hardware
        log.warning("Pico setup failed (%s) — no ring light / belt.", e)
        return None


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def run_calibration(
    cfg: dict,
    cap_params: dict,
    roi_cfg: Any,
    out_dir: Path,
    *,
    serial_cfg: dict | None = None,
    lights: bool = True,
    light_rgb=(180, 180, 180),
) -> int:
    """Grab one frame, run the detector, and write debug snapshots.

    Writes ``roi_snapshot.jpg`` and ``overlay.jpg`` (detected contour, bbox,
    centroid and the dedup band) to ``out_dir`` and logs ROI / area-fraction /
    circularity so you can tune ``camera.roi`` and ``capture.*`` thresholds in
    ``config.local.yaml``. The ring light is turned on so the snapshot matches
    capture conditions. ``out_dir`` lives under ``dataset.processed_dir`` so
    training never scans it as a class.
    """
    cam = cfg["camera"]
    width, height = int(cam["width"]), int(cam["height"])
    pico = maybe_open_pico(
        serial_cfg or {}, lights=lights, light_rgb=light_rgb,
        drive_belt=False, belt_speed_hz=0,
    )
    picam = _open_camera(
        width, height,
        cam.get("af_mode", "continuous"), cam.get("lens_position"),
        cam.get("awb", "auto"), cam.get("colour_gains"),
        cam.get("exposure_us"), cam.get("analogue_gain"),
    )
    try:
        for _ in range(5):  # let AE/AWB settle (under the ring light)
            frame = picam.capture_array()
            time.sleep(0.1)
    finally:
        pass

    roi_px = resolve_roi(roi_cfg, width, height)
    roi = crop_roi(frame, roi_px)
    roi_h, roi_w = roi.shape[:2]
    detector = CoinDetector(cap_params, roi_w, roi_h)
    det = detector.detect(roi)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "roi_snapshot.jpg"), roi)

    overlay = roi.copy()
    deduper = Deduper("centroid_band", cap_params, roi_w, roi_h)
    if deduper.travel_axis == "y":
        cv2.line(overlay, (0, int(deduper.band_lo)), (roi_w, int(deduper.band_lo)), (255, 0, 0), 1)
        cv2.line(overlay, (0, int(deduper.band_hi)), (roi_w, int(deduper.band_hi)), (255, 0, 0), 1)
    else:
        cv2.line(overlay, (int(deduper.band_lo), 0), (int(deduper.band_lo), roi_h), (255, 0, 0), 1)
        cv2.line(overlay, (int(deduper.band_hi), 0), (int(deduper.band_hi), roi_h), (255, 0, 0), 1)
    if det.found and det.bbox is not None and det.centroid is not None:
        x, y, w, h = det.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(overlay, det.centroid, 4, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)

    try:
        picam.stop()
    except Exception:  # pragma: no cover
        pass
    if pico is not None:
        try:
            if lights:
                pico.set_leds(0, 0, 0)  # ring off
        except Exception:  # pragma: no cover
            pass
        pico.close()

    roi_area = float(roi_w * roi_h)
    log.info("Calib: ROI px=%s in frame %dx%d", roi_px or (0, 0, width, height), width, height)
    if det.found:
        log.info(
            "Calib: coin found — area=%.0f area_frac=%.3f circularity=%.3f. "
            "Tune min/max_area_frac to bracket %.3f.",
            det.area,
            det.area / roi_area,
            det.circularity,
            det.area / roi_area,
        )
    else:
        log.info(
            "Calib: no coin detected. That's correct for empty belt; place a "
            "coin in the ROI and re-run to confirm a single clean blob."
        )
    log.info("Calib: snapshots written to %s (delete before zipping the dataset).", out_dir)
    return 0


# --------------------------------------------------------------------------- #
# Main capture loop
# --------------------------------------------------------------------------- #


def capture_loop(
    label: str,
    interval_s: float,
    count: int | None,
    preview: bool,
    out_root: Path,
    width: int,
    height: int,
    *,
    cap_params: dict | None = None,
    roi_cfg: Any = None,
    serial_cfg: dict | None = None,
    drive_belt: bool = False,
    belt_speed_hz: int = 800,
    lights: bool = True,
    light_rgb=(180, 180, 180),
    af_mode: str = "continuous",
    lens_position=None,
    awb_mode: str = "auto",
    colour_gains=None,
    exposure_us=None,
    analogue_gain=None,
) -> int:
    """Capture `count` images (or forever) to ``out_root/<label>/``.

    With gating on (``cap_params['gate']``), only coin-present frames are saved,
    tight-cropped, and ``count`` counts coins saved. With gating off, the
    ROI-cropped frame is saved every ``interval_s`` (legacy behaviour).
    Returns the number of images actually written.
    """
    cap_params = cap_params or {}
    gate = bool(cap_params.get("gate", True))

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing frames to %s (gate=%s)", out_dir, gate)

    picam = _open_camera(
        width, height, af_mode, lens_position, awb_mode, colour_gains,
        exposure_us, analogue_gain,
    )
    pico = maybe_open_pico(
        serial_cfg or {},
        lights=lights,
        light_rgb=light_rgb,
        drive_belt=drive_belt,
        belt_speed_hz=belt_speed_hz,
    )

    roi_px = resolve_roi(roi_cfg, width, height)
    roi_w = roi_px[2] if roi_px else width
    roi_h = roi_px[3] if roi_px else height

    detector = CoinDetector(cap_params, roi_w, roi_h) if gate else None
    deduper = (
        Deduper(str(cap_params.get("dedup_mode", "centroid_band")), cap_params, roi_w, roi_h)
        if gate
        else None
    )
    pad_frac = float(cap_params.get("crop_pad_frac", 0.25))
    crop_square = bool(cap_params.get("crop_square", True))

    written = 0
    try:
        if preview:
            try:
                from picamera2.previews import QtPreview  # type: ignore[import-not-found]

                picam.start_preview(QtPreview())
            except Exception as e:  # pragma: no cover - depends on host
                log.warning("Preview requested but unavailable: %s", e)

        while count is None or written < count:
            now = time.monotonic()
            frame = picam.capture_array()  # HxWx3, BGR-ordered (RGB888 config)
            roi = crop_roi(frame, roi_px)

            if not gate:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = out_dir / f"{ts}.jpg"
                cv2.imwrite(str(path), roi)
                written += 1
                log.info("[%d%s] %s", written, f"/{count}" if count else "", path.name)
                if count is None or written < count:
                    time.sleep(interval_s)
                continue

            det = detector.detect(roi)  # type: ignore[union-attr]
            if not deduper.should_save(det, now):  # type: ignore[union-attr]
                log.debug("skip found=%s area=%.0f circ=%.2f", det.found, det.area, det.circularity)
                time.sleep(0.005)  # keep the empty-belt loop from pegging a core
                continue

            crop = tight_square_crop(roi, det, pad_frac, crop_square)
            if crop is None:
                continue
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = out_dir / f"{ts}.jpg"
            cv2.imwrite(str(path), crop)
            written += 1
            log.info(
                "[%d%s] %s area_frac=%.3f circ=%.2f",
                written,
                f"/{count}" if count else "",
                path.name,
                det.area / float(roi_w * roi_h),
                det.circularity,
            )
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        try:
            picam.stop()
        except Exception:  # pragma: no cover
            pass
        if pico is not None:
            try:
                if drive_belt:
                    pico.stop()
                    pico.disable()
                if lights:
                    pico.set_leds(0, 0, 0)  # ring off
            except Exception:  # pragma: no cover
                pass
            pico.close()
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture labelled training images on the Pi.")
    p.add_argument("--label", required=True, help="Class label / folder name (e.g. penny).")
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of images to save (coins, with the gate on); omit for endless.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between saves when the gate is OFF (legacy / reject capture).",
    )
    p.add_argument("--preview", action="store_true", help="Show a live preview window.")
    p.add_argument(
        "--gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override capture.gate. --no-gate saves the ROI-cropped frame every --interval.",
    )
    p.add_argument("--method", choices=["brightness", "absdiff"], default=None, help="Override capture.method.")
    p.add_argument(
        "--dedup",
        choices=["centroid_band", "min_interval", "none"],
        default=None,
        help="Override capture.dedup_mode.",
    )
    p.add_argument(
        "--drive-belt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run the belt continuously via the Pico (RUN) while capturing.",
    )
    p.add_argument("--belt-speed", type=int, default=None, help="Belt step rate in Hz when --drive-belt.")
    p.add_argument(
        "--lights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Turn the WS2812 ring light on for the session (override capture.lights).",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Write ROI/detection debug snapshots and exit (for tuning roi + thresholds).",
    )
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    return p.parse_args(argv)


def _resolve_capture_params(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge config ``capture:`` with CLI overrides (CLI > config.local > config)."""
    cap = dict(cfg.get("capture", {}))
    if args.gate is not None:
        cap["gate"] = args.gate
    if args.method is not None:
        cap["method"] = args.method
    if args.dedup is not None:
        cap["dedup_mode"] = args.dedup

    # The reject class bypasses the single-coin gate (its junk/empty/cluster
    # frames would be discarded by the area+circularity filter). Honour an
    # explicit --gate, otherwise auto-disable.
    reject_label = str(cap.get("reject_label", "reject"))
    if args.label == reject_label and args.gate is None:
        cap["gate"] = False
        log.info("Label %r → gate bypassed. Feed debris / clusters / empty belt.", reject_label)
    return cap


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = _parse_args(argv)
    cfg = load_config(args.config)

    width = int(cfg["camera"]["width"])
    height = int(cfg["camera"]["height"])
    roi_cfg = cfg["camera"].get("roi")
    cap_params = _resolve_capture_params(cfg, args)

    lights = args.lights if args.lights is not None else bool(cap_params.get("lights", True))
    light_rgb = tuple(cap_params.get("light_rgb", (180, 180, 180)))

    if args.calibrate:
        processed = Path(cfg["dataset"].get("processed_dir", "data/processed"))
        return run_calibration(
            cfg, cap_params, roi_cfg, processed / "_calib",
            serial_cfg=cfg.get("serial"), lights=lights, light_rgb=light_rgb,
        )

    drive_belt = args.drive_belt if args.drive_belt is not None else bool(cap_params.get("drive_belt", False))
    belt_speed_hz = args.belt_speed if args.belt_speed is not None else int(cap_params.get("belt_speed_hz", 800))

    written = capture_loop(
        label=args.label,
        interval_s=args.interval,
        count=args.count,
        preview=args.preview,
        out_root=Path(cfg["dataset"]["raw_dir"]),
        width=width,
        height=height,
        cap_params=cap_params,
        roi_cfg=roi_cfg,
        serial_cfg=cfg.get("serial"),
        drive_belt=drive_belt,
        belt_speed_hz=belt_speed_hz,
        lights=lights,
        light_rgb=light_rgb,
        af_mode=cfg["camera"].get("af_mode", "continuous"),
        lens_position=cfg["camera"].get("lens_position"),
        awb_mode=cfg["camera"].get("awb", "auto"),
        colour_gains=cfg["camera"].get("colour_gains"),
        exposure_us=cfg["camera"].get("exposure_us"),
        analogue_gain=cfg["camera"].get("analogue_gain"),
    )
    log.info("Done. Wrote %d images.", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
