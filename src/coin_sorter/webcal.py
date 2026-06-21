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

log = logging.getLogger("coin_sorter.webcal")

# ----- shared state (one lock) --------------------------------------------- #
_lock = threading.Lock()
_frame_lock = threading.Lock()
_latest = [b""]
_running = [True]
_ctrl_ver = [0]  # bumped when a hardware control (focus/wb/ring/belt) changes
_state: dict = {}


def default_state(cfg: dict) -> dict:
    """Build the initial UI state from config (so the UI opens at current values)."""
    cam = cfg.get("camera", {})
    capc = cfg.get("capture", {})
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
        "session_active": False, "label": "", "count": 0,
        # live readout (written by the camera thread)
        "det": False, "area_frac": 0.0, "det_circ": 0.0,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def apply_setting(state: dict, key: str, val: str) -> bool:
    """Update one state key from a string value (parse + clamp). Pure / testable.

    Returns True if a hardware control (focus / white balance / ring / belt)
    changed, so the caller can bump the controls version.
    """
    try:
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
            state["feeder_hz"] = int(_clamp(float(val), 1, 5000))
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
        "method": st["method"], "blur_ksize": st["blur"],
        "min_area_frac": st["min_area"], "max_area_frac": st["max_area"],
        "min_circularity": st["circ"],
    }


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
        try:
            pico.set_leds(st["ring"], st["ring"], st["ring"])
            pico.run(st["hz"] * st.get("dir", 1)) if st["belt"] else pico.stop()
            (pico.feeder_run(st["feeder_hz"] * st.get("feeder_dir", 1))
             if st["feeder"] else pico.feeder_stop())
        except Exception as e:  # pragma: no cover - hardware
            log.warning("Pico control failed: %s", e)


def _draw_overlay(frame, roi_px, det, st: dict) -> None:  # type: ignore[no-untyped-def]
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
        ("DETECTED" if det.found else "no coin") + f"  area={det.area / area:.3f} circ={det.circularity:.2f}",
    ]
    if st["session_active"]:
        rows.append(f"REC {st['label']}: {st['count']}")
    for i, t in enumerate(rows):
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
    det_cache: dict = {"key": None, "det": None}
    applied_ver, prev_session = -1, False
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
                if det.found and dd.should_save(det, now):
                    crop = cap.tight_square_crop(roi_img, det, 0.25, True)
                    if crop is not None:
                        outdir = raw_dir / (st["label"] or "coin")
                        outdir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(str(outdir / f"{ts}.jpg"), crop)
                        with _lock:
                            _state["count"] += 1
            prev_session = st["session_active"]

            roi_area = float(rw * rh) or 1.0
            with _lock:
                _state["det"] = det.found
                _state["area_frac"] = det.area / roi_area
                _state["det_circ"] = det.circularity

            _draw_overlay(frame, roi_px, det, st)
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
</style></head><body>
<div class=wrap>
<div><img src="/stream.mjpg"><div id=stat>…</div></div>
<div class=panel>
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
 <label>speed</label><input id=feeder_hz type=range min=50 max=5000 step=25 oninput="sl('feeder_hz',this.value)"><span class=val id=feeder_hzv></span></div>
<h3>Capture</h3>
<div class=row><input id=label placeholder="label e.g. penny">
 <button id=recbtn onclick="toggleRec()">Start</button></div>
<div class=row id=recstat>idle</div>
<h3>Config</h3>
<div class=row><button onclick="showCfg()">Show config snippet</button></div>
<pre id=cfg></pre>
</div>
</div>
<script>
let belt=false, rec=false, dir=1, feeder=false, feederDir=1;
function set(k,v){fetch('/set?'+k+'='+encodeURIComponent(v));syncManual(k);}
function sl(k,v){document.getElementById(k+'v').textContent=(+v).toFixed(2);set(k,v);}
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
  else{fetch('/capture?action=stop');b.textContent='Start';b.classList.remove('active');}}
function showCfg(){fetch('/config').then(r=>r.text()).then(t=>document.getElementById('cfg').textContent=t);}
function init(s){for(const k of ['lens','red','blue','exp','gain','roi_x','roi_y','roi_w','roi_h','min_area','max_area','circ','ring','hz','feeder_hz']){
  const sk=(k==='exp')?'exp_us':k;
  const el=document.getElementById(k);if(el&&s[sk]!==undefined){el.value=s[sk];document.getElementById(k+'v').textContent=(+s[sk]).toFixed(2);}}
  document.getElementById('afauto').classList.toggle('active',s.focus_mode==='continuous');
  document.getElementById('wbauto').classList.toggle('active',s.awb_mode==='auto');
  document.getElementById('expauto').classList.toggle('active',s.exp_mode==='auto');
  if(s.dir!==undefined){dir=s.dir;const b=document.getElementById('dirbtn');
    b.textContent=dir>0?'Fwd':'Rev';b.classList.toggle('active',dir<0);}
  if(s.feeder_dir!==undefined){feederDir=s.feeder_dir;const b=document.getElementById('feeddirbtn');
    b.textContent=feederDir>0?'Fwd':'Rev';b.classList.toggle('active',feederDir<0);}}
fetch('/state').then(r=>r.json()).then(init);
setInterval(()=>{fetch('/status').then(r=>r.json()).then(s=>{
  document.getElementById('stat').textContent=(s.detected?'● COIN':'○ none')+
    '  area='+s.area_frac.toFixed(3)+'  circ='+s.circ.toFixed(2);
  document.getElementById('recstat').textContent=s.session?('REC '+s.label+': '+s.count):'idle';
});},500);
</script></body></html>
"""


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
                    "circ": round(_state["det_circ"], 3), "session": _state["session_active"],
                    "label": _state["label"], "count": _state["count"], "belt": _state["belt"],
                }
            self._send(200, "application/json", json.dumps(s).encode())
        elif u.path == "/state":
            with _lock:
                s = dict(_state)
            self._send(200, "application/json", json.dumps(s).encode())
        elif u.path == "/config":
            with _lock:
                snip = config_snippet(_state)
            self._send(200, "text/plain", snip.encode())
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
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    global _state
    _state = default_state(cfg)

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
