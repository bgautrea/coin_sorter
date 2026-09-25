"""Web calibration + capture UI for the belt camera (Pi-side, headless).

A single-process stdlib HTTP server that owns the camera and the Pico, streams a
live MJPEG preview with the coin-detection overlay, and exposes sliders to tune
focus, white balance, ROI and the gate thresholds — then lets you run belt-fed
capture of labelled training data straight from the browser. A "Show config"
button emits a ``config.local.yaml`` snippet so the tuned values lock into the
CLI tools.

    python -m coin_sorter.webcal            # then open http://<pi>:8080/

The web UI holds the camera exclusively — stop it before running
``coin_sorter.capture`` / ``--calibrate`` on the CLI.

Modelled on cam_test/stream.py; no extra dependencies (stdlib + cv2/numpy/
picamera2). picamera2/libcamera are imported only inside the camera thread, so
this module stays importable off-hardware.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import socketserver
import threading
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import cv2

from . import configure_logging, load_config
from . import capture as cap
from .sorter import DivertQueue

log = logging.getLogger("coin_sorter.webcal")

# ----- shared state (one lock) --------------------------------------------- #
_lock = threading.Lock()
_frame_lock = threading.Lock()
_latest = [b""]
_running = [True]
_ctrl_ver = [0]  # bumped when a hardware control (focus/wb/ring/belt) changes
_state: dict = {}
_capc: dict = {}  # config['capture'] — gate keys with no slider pass through as-is
_labels: list = []  # config['classifier']['labels'] — tag targets in the gallery
# Camera recipe from config (config.local.yaml on the rig). The "Recipe" button
# restores it after someone presses an Auto button or nudges a slider.
_RECIPE_KEYS = ("focus_mode", "lens", "awb_mode", "red", "blue", "exp_mode", "exp_us", "gain")
_recipe: dict = {}
_pico_lock = threading.Lock()  # camera thread and feeder pulser share the Pico
_raw_dir = [None]  # base directory for captured crops; set in main()
_clf = [None]  # CoinClassifier, loaded on the first sorting session
_cfg = [None]  # the loaded config, for routes that need it


def _get_classifier(cfg: dict):  # type: ignore[no-untyped-def]
    """Load the ONNX classifier once, on demand.

    Deferred so webcal still starts (for capture and tagging) on a machine with
    no trained model, and so the import cost is not paid by every session.
    """
    if _clf[0] is None:
        from .infer import CoinClassifier

        clf = CoinClassifier.from_config(cfg)
        if clf.is_stub:
            log.warning("No ONNX model — sorting would send everything to one bin.")
            return None
        log.info("Sorting with %s", ", ".join(clf.labels))
        _clf[0] = clf
    return _clf[0]


# Physical layout of the three cups, as seen standing at the discharge end.
# The firmware knows bins by name; the operator knows them by position.
# `recheck` is centre because centre is where the dish rests: a coin nothing
# aimed for lands there, and unexamined coins must be re-fed, not discarded.
BIN_SIDES = [("reject", "left"), ("recheck", "centre"), ("keep", "right")]
DEFAULT_BIN = "recheck"


def _default_bin_map(cfg: dict) -> dict:
    """label -> bin, seeded from sorter.sort_map so the UI opens at config."""
    sm = (cfg.get("sorter") or {}).get("sort_map") or {}
    out = {lab: b for b, labs in sm.items() for lab in (labs or [])}
    for lab in (cfg.get("classifier") or {}).get("labels") or []:
        out.setdefault(lab, DEFAULT_BIN)  # unmapped is a decision not to guess
    return out


def default_state(cfg: dict) -> dict:
    """Build the initial UI state from config (so the UI opens at current values)."""
    cam = cfg.get("camera", {})
    capc = cfg.get("capture", {})
    srt = cfg.get("sorter", {})
    roi = cam.get("roi") or [0.0, 0.0, 1.0, 1.0]
    gains = cam.get("colour_gains") or [2.0, 2.0]
    rgb = capc.get("light_rgb") or [180, 180, 180]
    return {
        "focus_mode": cam.get("af_mode", "continuous"),
        "lens": float(cam.get("lens_position") or 0.0),
        "awb_mode": cam.get("awb", "auto"),
        "red": float(gains[0]),
        "blue": float(gains[1]),
        "roi_x": float(roi[0]), "roi_y": float(roi[1]),
        "roi_w": float(roi[2]), "roi_h": float(roi[3]),
        "min_area": float(capc.get("min_area_frac", 0.02)),
        "max_area": float(capc.get("max_area_frac", 0.60)),
        "circ": float(capc.get("min_circularity", 0.65)),
        "blur": int(capc.get("blur_ksize", 5)),
        "method": capc.get("method", "brightness"),
        "dedup": capc.get("dedup_mode", "centroid_band"),
        "ring": int(rgb[0]),
        "exp_mode": "manual" if cam.get("exposure_us") else "auto",
        "exp_us": int(cam.get("exposure_us") or 4000),
        "gain": float(cam.get("analogue_gain") or 2.0),
        "belt": False,
        "dir": 1,  # 1 = forward, -1 = reverse
        "hz": int(capc.get("belt_speed_hz", 400)),
        "feeder": False,
        "feeder_dir": 1,  # 1 = forward, -1 = reverse
        "feeder_hz": int(capc.get("feeder_speed_hz", 400)),
        # pulsing: on for feeder_on_ms out of every feeder_period_ms (0 = continuous)
        "feeder_on_ms": int(capc.get("feeder_on_ms", 0)),
        "feeder_period_ms": int(capc.get("feeder_period_ms", 2000)),
        "session_active": False, "label": "", "count": 0,
        # Sorting session. bin_map is label -> bin; the UI shows it as
        # left/centre/right because that is what you see at the machine.
        "sort_active": False,
        "bin_map": dict(_default_bin_map(cfg)),
        "sort_counts": {b: 0 for b, _side in BIN_SIDES} | {"missed": 0},
        "in_flight": 0,
        "last_sort": "",
        "threshold": float((cfg.get("classifier") or {}).get("confidence_threshold", 0.9)),
        # Timing for the divert queue; see sorter.py. The camera decides far
        # upstream of the dish, so diverts are scheduled for arrival.
        "transport_delay_s": float(srt.get("transport_delay_s", 0.0)),
        "divert_lead_s": float(srt.get("divert_lead_s", 0.0)),
        "divert_hold_s": float(srt.get("divert_hold_s", 0.5)),
        # How long the dish stays on a bin after aiming. Late tolerance is
        # neutral_after_s - divert_lead_s, so this must exceed the lead or a
        # coin arriving even slightly late finds the dish already back at
        # centre and lands in the re-feed pile.
        "neutral_after_s": float(srt.get("neutral_after_s", 1.0)),
        # live readout (written by the camera thread)
        "det": False, "area_frac": 0.0, "det_circ": 0.0, "sharp": 0.0, "meta": {},
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def apply_setting(state: dict, key: str, val: str) -> bool:
    """Update one state key from a string value (parse + clamp). Pure / testable.

    Returns True if a hardware control (focus / white balance / ring / belt)
    changed, so the caller can bump the controls version.
    """
    try:
        if key == "preset":
            state.update(_recipe)
            return True
        if key == "focus_mode":
            state["focus_mode"] = "manual" if val == "manual" else "continuous"
            return True
        if key == "lens":
            state["lens"] = _clamp(float(val), 0.0, 15.0)
            state["focus_mode"] = "manual"
            return True
        if key == "awb_mode":
            state["awb_mode"] = "manual" if val == "manual" else "auto"
            return True
        if key in ("red", "blue"):
            state[key] = _clamp(float(val), 0.1, 8.0)
            state["awb_mode"] = "manual"
            return True
        if key == "exp_mode":
            state["exp_mode"] = "manual" if val == "manual" else "auto"
            return True
        if key == "exp":
            state["exp_us"] = int(_clamp(float(val), 50, 33000))
            state["exp_mode"] = "manual"
            return True
        if key == "gain":
            state["gain"] = _clamp(float(val), 1.0, 16.0)
            state["exp_mode"] = "manual"
            return True
        if key in ("roi_x", "roi_y", "roi_w", "roi_h"):
            state[key] = _clamp(float(val), 0.0, 1.0)
        elif key == "min_area":
            state["min_area"] = _clamp(float(val), 0.0, 1.0)
        elif key == "max_area":
            state["max_area"] = _clamp(float(val), 0.0, 1.0)
        elif key == "circ":
            state["circ"] = _clamp(float(val), 0.0, 1.0)
        elif key == "blur":
            state["blur"] = max(1, int(float(val)))
        elif key == "method":
            if val in ("brightness", "absdiff"):
                state["method"] = val
        elif key == "dedup":
            if val in ("centroid_band", "min_interval", "none"):
                state["dedup"] = val
        elif key == "ring":
            state["ring"] = int(_clamp(float(val), 0, 255))
            return True
        elif key == "belt":
            state["belt"] = val in ("1", "true", "True", "on")
            return True
        elif key == "dir":
            state["dir"] = -1 if val in ("-1", "rev", "reverse") else 1
            return True
        elif key == "hz":
            state["hz"] = int(_clamp(float(val), 1, 5000))
            return True
        elif key == "feeder":
            state["feeder"] = val in ("1", "true", "True", "on")
            return True
        elif key == "feeder_dir":
            state["feeder_dir"] = -1 if val in ("-1", "rev", "reverse") else 1
            return True
        elif key == "feeder_hz":
            state["feeder_hz"] = int(_clamp(float(val), 1, 10000))
            return True
        elif key == "feeder_on_ms":
            state["feeder_on_ms"] = int(_clamp(float(val), 0, 5000))
            return True
        elif key == "feeder_period_ms":
            state["feeder_period_ms"] = int(_clamp(float(val), 100, 10000))
            return True
    except (ValueError, TypeError):
        pass
    return False


def config_snippet(state: dict) -> str:
    """Emit a config.local.yaml snippet from the current state (pure / testable)."""
    lines = ["camera:"]
    lines.append(
        f"  roi: [{state['roi_x']:.3f}, {state['roi_y']:.3f}, "
        f"{state['roi_w']:.3f}, {state['roi_h']:.3f}]"
    )
    if state["focus_mode"] == "manual":
        lines += ["  af_mode: manual", f"  lens_position: {state['lens']:.2f}"]
    else:
        lines.append("  af_mode: continuous")
    if state["awb_mode"] == "manual":
        lines += ["  awb: manual", f"  colour_gains: [{state['red']:.2f}, {state['blue']:.2f}]"]
    else:
        lines.append("  awb: auto")
    if state["exp_mode"] == "manual":
        lines += [f"  exposure_us: {int(state['exp_us'])}", f"  analogue_gain: {state['gain']:.2f}"]
    lines.append("capture:")
    lines.append(f"  min_area_frac: {state['min_area']:.3f}")
    lines.append(f"  max_area_frac: {state['max_area']:.3f}")
    lines.append(f"  min_circularity: {state['circ']:.2f}")
    return "\n".join(lines) + "\n"


# ----- camera thread (owns camera + Pico) ---------------------------------- #


def _params(st: dict) -> dict:
    return {
        **_capc,  # min_fill_frac / reject_border / isolate / crop_pad_frac ...
        "method": st["method"], "blur_ksize": st["blur"],
        "min_area_frac": st["min_area"], "max_area_frac": st["max_area"],
        "min_circularity": st["circ"],
    }


def _pulsing(st: dict) -> bool:
    return bool(st["feeder"]) and 0 < st["feeder_on_ms"] < st["feeder_period_ms"]


def feeder_pulser(pico) -> None:  # type: ignore[no-untyped-def]
    """Run the feeder in bursts (feeder_on_ms of every feeder_period_ms).

    Owns the feeder while pulsing is enabled; _apply_hw leaves it alone then.
    Its own thread because FRUN blocks the Pico for the ~250 ms accel ramp and
    that must not stall the camera loop.
    """
    on_hw, t0 = False, time.monotonic()
    while _running[0]:
        with _lock:
            st = dict(_state)
        if not _pulsing(st):
            on_hw, t0 = False, time.monotonic()
            time.sleep(0.05)
            continue
        phase = ((time.monotonic() - t0) * 1000.0) % st["feeder_period_ms"]
        want = phase < st["feeder_on_ms"]
        if want != on_hw:
            with _pico_lock:
                try:
                    if want:
                        pico.feeder_run(st["feeder_hz"] * st.get("feeder_dir", 1))
                    else:
                        pico.feeder_stop()
                except Exception as e:  # pragma: no cover - hardware
                    log.warning("Feeder pulse failed: %s", e)
            on_hw = want
        time.sleep(0.02)


def _apply_hw(picam, pico, st: dict) -> None:  # type: ignore[no-untyped-def]
    cap.apply_autofocus(picam, st["focus_mode"], st["lens"])
    cap.apply_white_balance(
        picam, st["awb_mode"], (st["red"], st["blue"]) if st["awb_mode"] == "manual" else None
    )
    cap.apply_exposure(
        picam,
        st["exp_us"] if st["exp_mode"] == "manual" else None,
        st["gain"] if st["exp_mode"] == "manual" else None,
    )
    if pico is not None:
        with _pico_lock:
            try:
                pico.set_leds(st["ring"], st["ring"], st["ring"])
                pico.run(st["hz"] * st.get("dir", 1)) if st["belt"] else pico.stop()
                if not _pulsing(st):  # else the pulser thread owns the feeder
                    (pico.feeder_run(st["feeder_hz"] * st.get("feeder_dir", 1))
                     if st["feeder"] else pico.feeder_stop())
            except Exception as e:  # pragma: no cover - hardware
                log.warning("Pico control failed: %s", e)


def _draw_overlay(frame, roi_px, det, st: dict, sharp: float = 0.0, meta: dict | None = None) -> None:  # type: ignore[no-untyped-def]
    meta = meta or {}
    H, W = frame.shape[:2]
    x, y, w, h = roi_px if roi_px else (0, 0, W, H)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # dedup band (travel axis y, centre 0.5 +/- 0.08)
    b0, b1 = y + int(h * 0.42), y + int(h * 0.58)
    cv2.line(frame, (x, b0), (x + w, b0), (255, 120, 0), 1)
    cv2.line(frame, (x, b1), (x + w, b1), (255, 120, 0), 1)
    if det.found and det.bbox and det.centroid:
        bx, by, bw, bh = det.bbox
        cv2.rectangle(frame, (x + bx, y + by), (x + bx + bw, y + by + bh), (255, 255, 0), 2)
        cv2.circle(frame, (x + det.centroid[0], y + det.centroid[1]), 4, (0, 0, 255), -1)
    area = float(w * h) or 1.0
    rows = [
        "focus=" + st["focus_mode"] + (f" lens={st['lens']:.2f}" if st["focus_mode"] == "manual" else ""),
        "awb=" + st["awb_mode"] + (f" r={st['red']:.2f} b={st['blue']:.2f}" if st["awb_mode"] == "manual" else ""),
        "exp=" + st["exp_mode"] + (f" {st['exp_us']}us g={st['gain']:.1f}" if st["exp_mode"] == "manual" else ""),
        (f"sensor: {meta.get('exp_us', 0) / 1000:.1f}ms g={meta.get('gain', 0):.1f} "
         f"wb={meta.get('red', 0):.2f}/{meta.get('blue', 0):.2f} lux={meta.get('lux', 0):.0f}") if meta else "",
        ("DETECTED" if det.found else "no coin") + f"  area={det.area / area:.3f} circ={det.circularity:.2f} fill={det.fill:.2f}"
        + (f"  sharp={sharp:.0f}" if det.found else ""),
    ]
    if st["session_active"]:
        rows.append(f"REC {st['label']}: {st['count']}")
    for i, t in enumerate(t for t in rows if t):
        org = (x + 6, y + 24 + 24 * i)
        cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def camera_thread(cfg: dict, raw_dir: Path) -> None:
    cam = cfg["camera"]
    W, H = int(cam["width"]), int(cam["height"])
    with _lock:
        st = dict(_state)
    gains = (st["red"], st["blue"]) if st["awb_mode"] == "manual" else None
    picam = cap._open_camera(W, H, st["focus_mode"], st["lens"], st["awb_mode"], gains)
    pico = cap.maybe_open_pico(
        cfg.get("serial") or {}, lights=True, light_rgb=(st["ring"],) * 3,
        drive_belt=False, belt_speed_hz=st["hz"],
    )
    if pico is not None:
        threading.Thread(target=feeder_pulser, args=(pico,), daemon=True).start()
    det_cache: dict = {"key": None, "det": None}
    applied_ver, prev_session = -1, False
    prev_sort, clf, sort_dedup, queue = False, None, None, None
    dish_bin, last_divert = None, None
    frame_n, meta = 0, {}
    try:
        while _running[0]:
            now = time.monotonic()
            with _lock:
                st = dict(_state)
                ver = _ctrl_ver[0]
            if ver != applied_ver:
                _apply_hw(picam, pico, st)
                applied_ver = ver

            frame = picam.capture_array()  # BGR (RGB888 config)
            if frame_n % 10 == 0:
                # What the sensor actually applied — lets you lock auto's pick.
                try:
                    md = picam.capture_metadata()
                    cg = md.get("ColourGains") or (0.0, 0.0)
                    meta = {"exp_us": int(md.get("ExposureTime", 0)), "gain": round(float(md.get("AnalogueGain", 0)), 2),
                            "red": round(float(cg[0]), 2), "blue": round(float(cg[1]), 2),
                            "lux": round(float(md.get("Lux", 0)), 1)}
                except Exception:  # pragma: no cover - hardware
                    meta = {}
            frame_n += 1
            roi_px = cap.resolve_roi(
                [st["roi_x"], st["roi_y"], st["roi_w"], st["roi_h"]], W, H
            )
            roi_img = cap.crop_roi(frame, roi_px)
            rh, rw = roi_img.shape[:2]
            params = _params(st)
            key = (params["method"], params["blur_ksize"], params["min_area_frac"],
                   params["max_area_frac"], params["min_circularity"], rw, rh)
            if det_cache["key"] != key:
                det_cache["key"] = key
                det_cache["det"] = cap.CoinDetector(params, rw, rh)
            det = det_cache["det"].detect(roi_img)
            sharp = cap.coin_sharpness(roi_img, det)

            # capture session
            if st["session_active"]:
                if not prev_session:
                    self_dedup = cap.Deduper(
                        st["dedup"],
                        {**params, "travel_axis": "y", "band_center_frac": 0.5,
                         "band_halfwidth_frac": 0.08, "min_interval_s": 0.4},
                        rw, rh,
                    )
                    camera_thread._dedup = self_dedup  # type: ignore[attr-defined]
                dd = camera_thread._dedup  # type: ignore[attr-defined]
                # Called unconditionally: centroid_band re-arms on empty
                # frames from inside should_save (see the sorting block below).
                if dd.should_save(det, now):
                    crop = cap.tight_square_crop(
                        roi_img, det, float(_capc.get("crop_pad_frac", 0.25)),
                        bool(_capc.get("crop_square", True)),
                    )
                    if crop is not None:
                        outdir = raw_dir / (st["label"] or "coin")
                        outdir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(str(outdir / f"{ts}.jpg"), crop)
                        with _lock:
                            _state["count"] += 1
            prev_session = st["session_active"]

            # ---- sorting session -------------------------------------
            # Runs in this thread because webcal already owns the camera and
            # the Pico; a separate sorter process would fight it for both.
            if st["sort_active"]:
                if not prev_sort:
                    clf = _get_classifier(cfg)
                    sort_dedup = cap.Deduper(
                        "centroid_band",
                        {**params, "travel_axis": "y", "band_center_frac": 0.5,
                         "band_halfwidth_frac": 0.08, "min_interval_s": 0.4},
                        rw, rh,
                    )
                    queue = DivertQueue(
                        float(st["transport_delay_s"]), float(st["divert_hold_s"]),
                        float(st["divert_lead_s"]),
                    )
                # NB: should_save must be called on EVERY frame, including
                # empty ones -- centroid_band re-arms itself inside it when the
                # ROI goes empty. Short-circuiting on det.found leaves it
                # disarmed after the first coin and nothing sorts again.
                if clf is not None and sort_dedup.should_save(det, now):
                    crop = cap.tight_square_crop(
                        roi_img, det, float(_capc.get("crop_pad_frac", 0.25)),
                        bool(_capc.get("crop_square", True)),
                    )
                    if crop is not None:
                        label, conf = clf.predict(crop)
                        bin_ = st["bin_map"].get(label, DEFAULT_BIN)
                        if conf < float(st["threshold"]):
                            bin_ = DEFAULT_BIN   # cannot tell -> re-feed pile
                        queue.schedule(bin_, label, now)
                        with _lock:
                            _state["last_sort"] = f"{label} {conf:.2f} -> {bin_}"
                            _state["in_flight"] = len(queue)
                for bin_, label, late, missed in queue.due(now):
                    if missed:
                        with _lock:
                            _state["sort_counts"]["missed"] += 1
                        continue
                    if pico is not None:
                        try:
                            pico.sort(bin_)
                            dish_bin, last_divert = bin_, now
                        except Exception as e:  # pragma: no cover - hardware
                            log.error("SORT %s failed: %s", bin_, e)
                    with _lock:
                        _state["sort_counts"][bin_] = _state["sort_counts"].get(bin_, 0) + 1
                        _state["in_flight"] = len(queue)
                # park at neutral once the coin has cleared
                hold_for = float(st["neutral_after_s"])
                if (pico is not None and dish_bin not in (None, DEFAULT_BIN)
                        and last_divert and now - last_divert >= hold_for):
                    nxt = queue.next_due()
                    if nxt is None or nxt - now > hold_for:
                        try:
                            pico.sort(DEFAULT_BIN)
                            dish_bin, last_divert = DEFAULT_BIN, None
                        except Exception:  # pragma: no cover - hardware
                            pass
            prev_sort = st["sort_active"]

            roi_area = float(rw * rh) or 1.0
            with _lock:
                _state["det"] = det.found
                _state["area_frac"] = det.area / roi_area
                _state["det_circ"] = det.circularity
                _state["sharp"] = sharp
                _state["meta"] = meta

            _draw_overlay(frame, roi_px, det, st, sharp, meta)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _frame_lock:
                    _latest[0] = jpg.tobytes()
            time.sleep(0.01)
    finally:
        try:
            picam.stop()
        except Exception:  # pragma: no cover
            pass
        if pico is not None:
            with _pico_lock:
                for fn in (lambda: pico.stop(), lambda: pico.feeder_stop(),
                           lambda: pico.set_leds(0, 0, 0), pico.close):
                    try:
                        fn()
                    except Exception:  # pragma: no cover
                        pass


# ----- HTTP ----------------------------------------------------------------- #

PAGE = """<!doctype html><html><head><title>coin sorter — calibrate</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:10px}
.wrap{display:flex;flex-wrap:wrap;gap:12px}
img{max-width:100%;border:1px solid #333;background:#000}
.panel{background:#1c1c1c;padding:10px 14px;border-radius:8px;min-width:320px}
.row{margin:8px 0;display:flex;align-items:center;gap:8px}
.row label{width:90px;font-size:13px;color:#bbb}
input[type=range]{flex:1}
.val{width:54px;text-align:right;font-variant-numeric:tabular-nums;color:#9cf}
button{background:#444;color:#eee;border:1px solid #666;padding:6px 10px;border-radius:4px;cursor:pointer}
button.active{background:#0a6}
h3{margin:6px 0;font-size:14px;color:#8cf;border-bottom:1px solid #333}
#stat{font-family:monospace;font-size:13px;color:#9f9}
pre{background:#000;padding:8px;border-radius:4px;white-space:pre-wrap;font-size:12px}
input#label{background:#222;color:#eee;border:1px solid #555;padding:5px;border-radius:4px;width:120px}
select#gallabel{background:#222;color:#eee;border:1px solid #555;padding:5px;border-radius:4px;max-width:230px}
.gallery{display:flex;flex-wrap:wrap;gap:6px;max-height:460px;overflow:auto;margin-top:8px}
#tagger{display:none;margin-top:8px;padding:8px;border:1px solid #444;border-radius:6px;background:#181818}
#tagimg{width:320px;height:320px;object-fit:contain;background:#000;border-radius:4px;display:block}
#tagbtns{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
#tagbtns button{font-size:12px;padding:4px 8px}
#tagbtns button b{color:#9cf;margin-right:4px}
.thumb.sel img{outline:2px solid #0a6}
.thumb{position:relative;width:92px;height:92px}
.thumb img{width:92px;height:92px;object-fit:cover;border:1px solid #333;border-radius:4px;cursor:zoom-in}
.thumb button{position:absolute;top:2px;right:2px;width:20px;height:20px;padding:0;line-height:18px;
  font-size:12px;background:#a22;border:1px solid #c44;border-radius:3px;opacity:.85;cursor:pointer}
.thumb button:hover{opacity:1;background:#e33}
</style></head><body>
<div class=wrap>
<div><img src="/stream.mjpg"><div id=stat>…</div></div>
<div class=panel>
<h3>Camera</h3>
<div class=row><button id=recipe onclick="applyRecipe()" style="background:#2a5">Recipe</button>
 <span style="font-size:12px;color:#999">restore the calibrated focus / WB / exposure from config</span></div>
<h3>Focus</h3>
<div class=row><button id=afauto onclick="set('focus_mode','continuous')">Auto</button>
 <label>lens</label><input id=lens type=range min=0 max=15 step=0.05 oninput="sl('lens',this.value)">
 <span class=val id=lensv></span></div>
<h3>White balance</h3>
<div class=row><button id=wbauto onclick="set('awb_mode','auto')">Auto</button></div>
<div class=row><label>red</label><input id=red type=range min=0.5 max=5 step=0.02 oninput="sl('red',this.value)"><span class=val id=redv></span></div>
<div class=row><label>blue</label><input id=blue type=range min=0.5 max=5 step=0.02 oninput="sl('blue',this.value)"><span class=val id=bluev></span></div>
<h3>Exposure</h3>
<div class=row><button id=expauto onclick="set('exp_mode','auto')">Auto</button>
 <span style="font-size:12px;color:#999">shorter = freezes motion</span></div>
<div class=row><label>shutter &micro;s</label><input id=exp type=range min=100 max=20000 step=100 oninput="sl('exp',this.value)"><span class=val id=expv></span></div>
<div class=row><label>gain</label><input id=gain type=range min=1 max=16 step=0.5 oninput="sl('gain',this.value)"><span class=val id=gainv></span></div>
<h3>ROI (fractions)</h3>
<div class=row><label>x</label><input id=roi_x type=range min=0 max=1 step=0.005 oninput="sl('roi_x',this.value)"><span class=val id=roi_xv></span></div>
<div class=row><label>y</label><input id=roi_y type=range min=0 max=1 step=0.005 oninput="sl('roi_y',this.value)"><span class=val id=roi_yv></span></div>
<div class=row><label>w</label><input id=roi_w type=range min=0.02 max=1 step=0.005 oninput="sl('roi_w',this.value)"><span class=val id=roi_wv></span></div>
<div class=row><label>h</label><input id=roi_h type=range min=0.02 max=1 step=0.005 oninput="sl('roi_h',this.value)"><span class=val id=roi_hv></span></div>
<h3>Gate</h3>
<div class=row><label>min area</label><input id=min_area type=range min=0 max=0.3 step=0.002 oninput="sl('min_area',this.value)"><span class=val id=min_areav></span></div>
<div class=row><label>max area</label><input id=max_area type=range min=0.05 max=1 step=0.01 oninput="sl('max_area',this.value)"><span class=val id=max_areav></span></div>
<div class=row><label>circularity</label><input id=circ type=range min=0 max=1 step=0.01 oninput="sl('circ',this.value)"><span class=val id=circv></span></div>
</div>
<div class=panel>
<h3>Ring light</h3>
<div class=row><label>brightness</label><input id=ring type=range min=0 max=255 step=1 oninput="sl('ring',this.value)"><span class=val id=ringv></span></div>
<h3>Belt</h3>
<div class=row><button id=beltbtn onclick="toggleBelt()">Run</button>
 <button id=dirbtn onclick="toggleDir()">Fwd</button>
 <label>speed</label><input id=hz type=range min=50 max=1500 step=10 oninput="sl('hz',this.value)"><span class=val id=hzv></span></div>
<h3>Feeder</h3>
<div class=row><button id=feedbtn onclick="toggleFeeder()">Run</button>
 <button id=feeddirbtn onclick="toggleFeederDir()">Fwd</button>
 <label>speed</label><input id=feeder_hz type=range min=50 max=10000 step=50 oninput="sl('feeder_hz',this.value)"><span class=val id=feeder_hzv></span></div>
<div class=row><label>burst ms (0=continuous)</label><input id=feeder_on_ms type=range min=0 max=3000 step=50 oninput="sl('feeder_on_ms',this.value)"><span class=val id=feeder_on_msv></span>
 <label>every ms</label><input id=feeder_period_ms type=range min=200 max=10000 step=100 oninput="sl('feeder_period_ms',this.value)"><span class=val id=feeder_period_msv></span></div>
<h3>Capture</h3>
<div class=row><input id=label placeholder="label e.g. penny">
 <button id=recbtn onclick="toggleRec()">Start</button></div>
<div class=row id=recstat>idle</div>
<h3>Config</h3>
<div class=row><button onclick="showCfg()">Show config snippet</button></div>
<pre id=cfg></pre>
</div>
<div class=panel style="min-width:340px">
<h3>Captures</h3>
<div class=row>
 <select id=gallabel onchange="galOffset=0;loadCaptures()"></select>
 <button onclick="loadFolders().then(loadCaptures)">Refresh</button>
 <button onclick="galPage(-1)">&lsaquo; Prev</button>
 <button onclick="galPage(1)">Next &rsaquo;</button>
 <span id=galcount style="font-size:12px;color:#9cf"></span>
 <button onclick="startTag()">Tag</button>
</div>
<div id=tagger>
 <div style="display:flex;gap:10px;align-items:flex-start">
  <img id=tagimg>
  <div style="font-size:12px;color:#aaa;max-width:260px">
   <div id=tagname style="font-family:monospace;color:#9cf"></div>
   <div id=tagpos></div>
   <p>Press a key to move this crop into that label and advance.
   <b>s</b> skip &nbsp; <b>u</b> undo last &nbsp; <b>x</b> delete &nbsp; <b>Esc</b> stop</p>
   <div id=tagbtns></div>
  </div>
 </div>
</div>
<div id=sortpanel style="border:1px solid #333;padding:8px;margin:8px 0">
 <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
  <button id=sortbtn onclick="toggleSort()">Start sorting</button>
  <span id=sortmodel style="font-size:11px;color:#888"></span>
  <span id=sortstat style="font-size:12px;color:#9cf"></span>
 </div>
 <div id=sortmap style="margin-top:8px"></div>
 <p style="font-size:11px;color:#888;margin:6px 0 0">
  Each class goes to one cup. Anything the model scores below the confidence
  threshold goes to <b>check</b> regardless, so "cannot tell" and "not mapped"
  land in the same re-feed pile. Coins divert about
  <span id=sortdelay></span>s after they pass the camera, so the cups fill on a
  delay -- let the belt run that long past your last coin.</p>
</div>
<div id=gallery class=gallery></div>
</div>
</div>
<script>
let belt=false, rec=false, dir=1, feeder=false, feederDir=1;
function set(k,v){fetch('/set?'+k+'='+encodeURIComponent(v));syncManual(k);}
function sl(k,v){document.getElementById(k+'v').textContent=(+v).toFixed(2);set(k,v);}
function applyRecipe(){fetch('/set?preset=1').then(()=>fetch('/state').then(r=>r.json()).then(init));}
function syncManual(k){
  if(k==='lens')document.getElementById('afauto').classList.remove('active');
  if(k==='focus_mode')document.getElementById('afauto').classList.add('active');
  if(k==='red'||k==='blue')document.getElementById('wbauto').classList.remove('active');
  if(k==='awb_mode')document.getElementById('wbauto').classList.add('active');
  if(k==='exp'||k==='gain')document.getElementById('expauto').classList.remove('active');
  if(k==='exp_mode')document.getElementById('expauto').classList.add('active');
}
function toggleBelt(){belt=!belt;document.getElementById('beltbtn').classList.toggle('active',belt);
  document.getElementById('beltbtn').textContent=belt?'Stop':'Run';fetch('/set?belt='+(belt?1:0));}
function toggleDir(){dir=-dir;const b=document.getElementById('dirbtn');
  b.textContent=dir>0?'Fwd':'Rev';b.classList.toggle('active',dir<0);fetch('/set?dir='+dir);}
function toggleFeeder(){feeder=!feeder;document.getElementById('feedbtn').classList.toggle('active',feeder);
  document.getElementById('feedbtn').textContent=feeder?'Stop':'Run';fetch('/set?feeder='+(feeder?1:0));}
function toggleFeederDir(){feederDir=-feederDir;const b=document.getElementById('feeddirbtn');
  b.textContent=feederDir>0?'Fwd':'Rev';b.classList.toggle('active',feederDir<0);fetch('/set?feeder_dir='+feederDir);}
function toggleRec(){rec=!rec;const b=document.getElementById('recbtn');
  if(rec){const l=document.getElementById('label').value||'coin';
    fetch('/capture?action=start&label='+encodeURIComponent(l));b.textContent='Stop';b.classList.add('active');}
  else{fetch('/capture?action=stop');b.textContent='Start';b.classList.remove('active');
    galOffset=0;loadFolders().then(()=>{const g=document.getElementById('gallabel'),l=document.getElementById('label').value;
      if([...g.options].some(o=>o.value===l))g.value=l;loadCaptures();});}}
function showCfg(){fetch('/config').then(r=>r.text()).then(t=>document.getElementById('cfg').textContent=t);}
let galOffset=0;
function loadFolders(){
  const sel=document.getElementById('gallabel'),cur=sel.value;
  return fetch('/folders').then(r=>r.json()).then(d=>{
    sel.innerHTML='';
    const on=new Set(d.folders.map(f=>f.name));
    for(const f of d.folders){const o=document.createElement('option');o.value=f.name;o.textContent=f.name+' ('+f.count+')';sel.appendChild(o);}
    for(const l of d.labels){if(on.has(l))continue;const o=document.createElement('option');o.value=l;o.textContent=l+' (empty)';sel.appendChild(o);}
    if(cur&&[...sel.options].some(o=>o.value===cur))sel.value=cur;
  });
}
function galPage(dir){galOffset=Math.max(0,galOffset+dir*80);loadCaptures();}
function loadCaptures(){
  const l=document.getElementById('gallabel').value;if(!l)return;
  fetch('/captures?label='+encodeURIComponent(l)+'&offset='+galOffset).then(r=>r.json()).then(d=>{
    if(d.offset>0&&!d.files.length){galOffset=Math.max(0,galOffset-80);loadCaptures();return;}
    document.getElementById('galcount').textContent=d.files.length?
      (d.offset+1)+'–'+(d.offset+d.files.length)+' of '+d.total+' (newest first)':'0 of '+d.total;
    const g=document.getElementById('gallery');g.innerHTML='';
    for(const n of d.files){
      const div=document.createElement('div');div.className='thumb';
      const img=document.createElement('img');img.title=n;
      img.src='/capture_img?label='+encodeURIComponent(l)+'&name='+encodeURIComponent(n);
      img.onclick=()=>window.open(img.src,'_blank');
      const b=document.createElement('button');b.textContent='✕';b.title='delete';
      b.onclick=()=>delCap(l,n,div);
      div.appendChild(img);div.appendChild(b);g.appendChild(div);tagDiv[n]=div;
    }
  });
}
let LABELS=[], tagQ=[], tagDiv={}, tagOn=false, tagUndo=[];
function tagTargets(l){const pre=l.split('_')[0];const t=LABELS.filter(x=>x.startsWith(pre+'_'));return t.length?t:LABELS;}
function tagKey(i){return i<9?String(i+1):String.fromCharCode(97+i-9);}
function startTag(){
  const l=document.getElementById('gallabel').value;if(!l)return;
  tagQ=[...document.querySelectorAll('#gallery .thumb img')].map(i=>i.title);
  if(!tagQ.length){loadCaptures();return;}
  const T=tagTargets(l),bt=document.getElementById('tagbtns');bt.innerHTML='';
  T.forEach((t,i)=>{const b=document.createElement('button');b.innerHTML='<b>'+tagKey(i)+'</b>'+t;
    b.onclick=()=>tagTo(t);bt.appendChild(b);});
  tagOn=true;document.getElementById('tagger').style.display='block';showTag();
}
function showTag(){
  const l=document.getElementById('gallabel').value;
  document.querySelectorAll('#gallery .thumb.sel').forEach(d=>d.classList.remove('sel'));
  if(!tagQ.length){stopTag();loadCaptures();return;}
  const n=tagQ[0];document.getElementById('tagimg').src='/capture_img?label='+encodeURIComponent(l)+'&name='+encodeURIComponent(n);
  document.getElementById('tagname').textContent=n;
  document.getElementById('tagpos').textContent=tagQ.length+' left in this page';
  const d=tagDiv[n];if(d){d.classList.add('sel');d.scrollIntoView({block:'nearest'});}
}
function tagTo(t){
  const l=document.getElementById('gallabel').value,n=tagQ.shift();
  fetch('/capture_move?label='+encodeURIComponent(l)+'&name='+encodeURIComponent(n)+'&to='+encodeURIComponent(t),{method:'POST'})
    .then(r=>{if(r.ok){tagUndo.push({from:l,name:n,to:t});if(tagDiv[n])tagDiv[n].hidden=true;}});
  showTag();
}
function tagUndoLast(){
  const u=tagUndo.pop();if(!u)return;
  fetch('/capture_move?label='+encodeURIComponent(u.to)+'&name='+encodeURIComponent(u.name)+'&to='+encodeURIComponent(u.from),{method:'POST'})
    .then(r=>{if(r.ok){tagQ.unshift(u.name);if(tagDiv[u.name])tagDiv[u.name].hidden=false;showTag();}});
}
function stopTag(){tagOn=false;document.getElementById('tagger').style.display='none';loadFolders();}
document.addEventListener('keydown',e=>{
  if(!tagOn||e.target.tagName==='INPUT')return;
  if(e.key==='Escape'){stopTag();return;}
  if(e.key==='s'){tagQ.shift();showTag();return;}
  if(e.key==='u'){tagUndoLast();return;}
  if(e.key==='x'){const l=document.getElementById('gallabel').value,n=tagQ.shift();delCap(l,n,tagDiv[n]);showTag();return;}
  const T=tagTargets(document.getElementById('gallabel').value),i=T.findIndex((_,i)=>tagKey(i)===e.key);
  if(i>=0)tagTo(T[i]);
});
function delCap(l,n,div){
  fetch('/capture_del?label='+encodeURIComponent(l)+'&name='+encodeURIComponent(n),{method:'POST'})
    .then(r=>{if(r.ok&&div)div.remove();});
}
function renderSortMap(){
  return fetch('/sortlabels').then(r=>r.json()).then(d=>{
    document.getElementById('sortmodel').textContent =
      d.stub ? 'no model loaded' : d.model;
    const box=document.getElementById('sortmap'); box.innerHTML='';
    for(const lab of d.labels){
      const row=document.createElement('div');
      row.style.cssText='display:flex;gap:6px;align-items:center;margin:3px 0';
      const n=document.createElement('span');
      n.textContent=lab; n.style.cssText='width:150px;font-family:monospace;font-size:12px';
      row.appendChild(n);
      for(const s of d.sides){
        const b=document.createElement('button');
        b.textContent=s.side;
        if(d.bin_map[lab]===s.bin)b.classList.add('active');
        b.onclick=()=>{fetch('/sortmap?label='+encodeURIComponent(lab)+'&bin='+s.bin)
          .then(()=>renderSortMap());};
        row.appendChild(b);
      }
      box.appendChild(row);
    }
  });
}
function toggleSort(){
  const on=document.getElementById('sortbtn').classList.contains('active');
  fetch('/sort?action='+(on?'stop':'start')).then(r=>r.json()).then(paintSort);
}
function paintSort(s){
  const b=document.getElementById('sortbtn');
  if(s.sort_active){b.classList.add('active');b.textContent='Stop sorting';}
  else{b.classList.remove('active');b.textContent='Start sorting';}
  const c=s.sort_counts||{};
  document.getElementById('sortstat').textContent=
    'left '+(c.reject||0)+'  centre '+(c.recheck||0)+'  right '+(c.keep||0)+
    (c.missed?('  missed '+c.missed):'')+
    '   in flight '+(s.in_flight||0)+(s.last_sort?('   last: '+s.last_sort):'');
}
function init(s){LABELS=s.labels||[];
  document.getElementById('sortdelay').textContent=
    ((s.transport_delay_s||0)-(s.divert_lead_s||0)).toFixed(1);
  renderSortMap(); paintSort(s);for(const k of ['lens','red','blue','exp','gain','roi_x','roi_y','roi_w','roi_h','min_area','max_area','circ','ring','hz','feeder_hz','feeder_on_ms','feeder_period_ms']){
  const sk=(k==='exp')?'exp_us':k;
  const el=document.getElementById(k);if(el&&s[sk]!==undefined){el.value=s[sk];document.getElementById(k+'v').textContent=(+s[sk]).toFixed(2);}}
  document.getElementById('afauto').classList.toggle('active',s.focus_mode==='continuous');
  document.getElementById('wbauto').classList.toggle('active',s.awb_mode==='auto');
  document.getElementById('expauto').classList.toggle('active',s.exp_mode==='auto');
  if(s.dir!==undefined){dir=s.dir;const b=document.getElementById('dirbtn');
    b.textContent=dir>0?'Fwd':'Rev';b.classList.toggle('active',dir<0);}
  if(s.feeder_dir!==undefined){feederDir=s.feeder_dir;const b=document.getElementById('feeddirbtn');
    b.textContent=feederDir>0?'Fwd':'Rev';b.classList.toggle('active',feederDir<0);}
  loadFolders().then(()=>{const g=document.getElementById('gallabel');
    if(s.label&&[...g.options].some(o=>o.value===s.label))g.value=s.label;loadCaptures();});}
fetch('/state').then(r=>r.json()).then(init);
setInterval(()=>{fetch('/status').then(r=>r.json()).then(s=>{
  document.getElementById('stat').textContent=(s.detected?'● COIN':'○ none')+
    '  area='+s.area_frac.toFixed(3)+'  circ='+s.circ.toFixed(2)+(s.detected?'  sharp='+s.sharp.toFixed(0):'');
  document.getElementById('recstat').textContent=s.session?('REC '+s.label+': '+s.count):'idle';
  paintSort(s);
});},500);
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# Capture browsing / deletion
# --------------------------------------------------------------------------- #


def _label_dir(label: str):
    """Resolve the capture directory for ``label``, or None if unsafe/unset.

    ``Path(...).name`` strips any directory components, so a crafted label like
    ``../foo`` cannot escape the raw-capture root.
    """
    base = _raw_dir[0]
    if base is None:
        return None
    name = Path(label or "").name
    if not name:
        return None
    return Path(base) / name


def list_captures(label: str, limit: int = 80, offset: int = 0):
    """Return (newest-first filenames for one page, total count) for ``label``."""
    d = _label_dir(label)
    if d is None or not d.is_dir():
        return [], 0
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files[offset:offset + limit]], len(files)


def _capture_file(label: str, name: str):
    """Resolve a single capture file path, guarding against path traversal."""
    d = _label_dir(label)
    if d is None:
        return None
    fname = Path(name or "").name
    if not fname.lower().endswith(".jpg"):
        return None
    return d / fname


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            self._send(200, "text/html", PAGE.encode())
        elif u.path == "/set":
            hw = False
            with _lock:
                for k, vs in q.items():
                    if apply_setting(_state, k, vs[0]):
                        hw = True
                if hw:
                    _ctrl_ver[0] += 1
            self.send_response(204)
            self.end_headers()
        elif u.path == "/capture":
            action = q.get("action", [""])[0]
            with _lock:
                if action == "start":
                    _state["label"] = (q.get("label", ["coin"])[0] or "coin").strip()
                    _state["count"] = 0
                    _state["session_active"] = True
                elif action == "stop":
                    _state["session_active"] = False
            self.send_response(204)
            self.end_headers()
        elif u.path == "/status":
            with _lock:
                s = {
                    "detected": _state["det"], "area_frac": round(_state["area_frac"], 4),
                    "circ": round(_state["det_circ"], 3), "sharp": round(_state["sharp"], 1),
                    "meta": _state["meta"],
                    "session": _state["session_active"],
                    "label": _state["label"], "count": _state["count"], "belt": _state["belt"],
                    "sort_active": _state["sort_active"],
                    "sort_counts": dict(_state["sort_counts"]),
                    "in_flight": _state["in_flight"], "last_sort": _state["last_sort"],
                }
            self._send(200, "application/json", json.dumps(s).encode())
        elif u.path == "/state":
            with _lock:
                s = dict(_state)
            # Tagging targets are the label TAXONOMY, which is not the same as
            # the deployed model's output classes. config.local.yaml overrides
            # classifier.labels to whatever the current ONNX emits (for v0,
            # just dime/nickel/penny/reject), and tagTargets() filters by
            # prefix -- so retagging a `nickel` crop into nickel_buffalo_obv
            # offered no targets at all. Union in whatever exists on disk.
            base = _raw_dir[0]
            on_disk = (
                [d.name for d in base.iterdir()
                 if d.is_dir() and not d.name.startswith(("_", "."))]
                if base is not None and base.is_dir() else []
            )
            s["labels"] = sorted(set(_labels) | set(on_disk))
            self._send(200, "application/json", json.dumps(s).encode())
        elif u.path == "/config":
            with _lock:
                snip = config_snippet(_state)
            self._send(200, "text/plain", snip.encode())
        elif u.path == "/sort":
            act = q.get("action", ["status"])[0]
            with _lock:
                if act == "start":
                    _state["sort_active"] = True
                    _state["sort_counts"] = {b: 0 for b, _s in BIN_SIDES} | {"missed": 0}
                elif act == "stop":
                    _state["sort_active"] = False
                s = {k: _state[k] for k in
                     ("sort_active", "sort_counts", "in_flight", "last_sort", "bin_map")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif u.path == "/sortmap":
            lab = q.get("label", [""])[0]
            b = q.get("bin", [""])[0]
            ok = bool(lab) and b in {name for name, _side in BIN_SIDES}
            if ok:
                with _lock:
                    _state["bin_map"][lab] = b
            self._send(200, "application/json",
                       json.dumps({"ok": ok, "label": lab, "bin": b}).encode())
        elif u.path == "/sortlabels":
            # Labels the loaded model actually emits, so the UI cannot offer a
            # mapping for a class this model will never predict.
            clf = _get_classifier(_cfg[0]) if _cfg[0] else None
            with _lock:
                bm = dict(_state["bin_map"])
            labs = list(clf.labels) if clf is not None else sorted(bm)
            self._send(200, "application/json", json.dumps({
                "labels": labs, "bin_map": bm,
                "sides": [{"bin": b, "side": sd} for b, sd in BIN_SIDES],
                "model": (_cfg[0].get("model", {}) or {}).get("path", "") if _cfg[0] else "",
                "stub": clf is None,
            }).encode())
        elif u.path == "/folders":
            base = _raw_dir[0]
            out = []
            if base is not None and base.is_dir():
                for d in sorted(base.iterdir()):
                    if d.is_dir() and not d.name.startswith(("_", ".")):
                        out.append({"name": d.name, "count": sum(1 for p in d.iterdir() if p.suffix.lower() == ".jpg")})
            self._send(200, "application/json", json.dumps({"folders": out, "labels": list(_labels)}).encode())
        elif u.path == "/captures":
            label = q.get("label", [_state.get("label", "")])[0]
            offset = max(0, int(q.get("offset", ["0"])[0] or 0))
            names, total = list_captures(label, 80, offset)
            self._send(200, "application/json",
                       json.dumps({"label": label, "files": names, "total": total, "offset": offset}).encode())
        elif u.path == "/capture_img":
            p = _capture_file(q.get("label", [""])[0], q.get("name", [""])[0])
            if p is not None and p.is_file():
                self._send(200, "image/jpeg", p.read_bytes())
            else:
                self.send_error(404)
        elif u.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            last = None
            try:
                while _running[0]:
                    with _frame_lock:
                        f = _latest[0]
                    if f and f is not last:
                        last = f
                        self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                        self.wfile.write(str(len(f)).encode())
                        self.wfile.write(b"\r\n\r\n")
                        self.wfile.write(f)
                        self.wfile.write(b"\r\n")
                    else:
                        time.sleep(1.0 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/capture_move":
            # Tag a crop: move it into another label folder (obv/rev tagging,
            # or fixing a mis-fed coin). Target must be a configured label.
            p = _capture_file(q.get("label", [""])[0], q.get("name", [""])[0])
            to = q.get("to", [""])[0]
            dst = _label_dir(to)
            if dst is not None and to not in _labels and not dst.is_dir():
                dst = None  # new folders only for configured labels; undo may target any existing one
            ok = False
            if p is not None and p.is_file() and dst is not None:
                try:
                    dst.mkdir(parents=True, exist_ok=True)
                    p.rename(dst / p.name)
                    ok = True
                except OSError as e:
                    log.warning("move failed: %s", e)
            self.send_response(204 if ok else 400)
            self.end_headers()
        elif u.path == "/capture_del":
            p = _capture_file(q.get("label", [""])[0], q.get("name", [""])[0])
            ok = False
            if p is not None and p.is_file():
                try:
                    p.unlink()
                    ok = True
                except OSError as e:
                    log.warning("delete failed: %s", e)
            self.send_response(204 if ok else 404)
            self.end_headers()
        else:
            self.send_error(404)


class _ThreadedHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(description="Web calibration + capture UI for the belt camera.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    _cfg[0] = cfg
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    _raw_dir[0] = raw_dir
    global _state
    _state = default_state(cfg)
    _capc.update(cfg.get("capture") or {})
    _recipe.update({k: _state[k] for k in _RECIPE_KEYS})
    _labels.extend((cfg.get("classifier") or {}).get("labels") or [])

    cam = threading.Thread(target=camera_thread, args=(cfg, raw_dir), daemon=True)
    cam.start()
    httpd = _ThreadedHTTP(("0.0.0.0", args.port), _Handler)
    log.info("Web calibration UI on http://0.0.0.0:%d/  (Ctrl-C to stop)", args.port)
    log.info("Note: this owns the camera — stop it before CLI capture/calibrate.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        _running[0] = False
        time.sleep(0.3)
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
