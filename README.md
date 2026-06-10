# Coin Sorter — Computer Vision Pipeline

Real-time coin classifier for an automated sorter. Camera on a Raspberry Pi 5
watches a belt, an ONNX classifier identifies each coin (penny / nickel / dime
/ quarter / reject), and a serial command tells a Raspberry Pi Pico to route
the coin to the correct bin.

```
   CSI camera ──► Pi 5 ──► picamera2 ──► OpenCV ──► ONNX Runtime ──► label, conf
                                                                       │
                                                  USB serial @ 115200 ─┴──► Pico ──► diverter
```

Training happens off-Pi (Google Colab, T4). The Pi never installs torch or
ultralytics — only `onnxruntime` for inference.

## Repo layout

```
coin_sorter/
├── pyproject.toml
├── config.yaml
├── src/coin_sorter/
│   ├── __init__.py    # load_config, configure_logging
│   ├── capture.py     # CLI: capture labelled frames on the Pi
│   ├── dataset.py     # PyTorch Dataset used by the Colab notebook
│   ├── train.py       # reference CLI mirror of the notebook
│   ├── infer.py       # CoinClassifier — ONNX Runtime, with stub fallback
│   ├── pico.py        # serial wrapper for the Pico belt controller
│   └── sorter.py      # main runtime loop
├── notebooks/train_colab.ipynb
├── scripts/zip_dataset.sh, fetch_model.sh
├── data/raw/, data/processed/   (gitignored)
├── models/                       (gitignored)
└── tests/
```

## Pi-side setup (Raspberry Pi OS Bookworm)

### 1. System packages

```bash
sudo apt update
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    git \
    zip
```

`python3-picamera2` brings in `libcamera` and the Pi-specific camera stack —
those bits are not available from PyPI on the Pi and must come from apt.

### 2. Virtual environment (with system site packages)

Bookworm enforces [PEP 668](https://peps.python.org/pep-0668/) — pip into the
system Python is blocked, and you should not use `--break-system-packages`.
Use a venv. **Critical:** `picamera2` has C extensions tied to the system
`libcamera` build, so the venv has to inherit system site-packages:

```bash
cd ~/coin_sorter
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Verify picamera2 is visible inside the venv:

```bash
python -c "from picamera2 import Picamera2; print('ok')"
```

If you forgot `--system-site-packages`, recreate the venv — installing
picamera2 from PyPI on the Pi is more pain than it is worth.

### 3. Confirm camera + Pico

```bash
libcamera-hello --list-cameras   # should show the Module 3
ls /dev/ttyACM*                  # should show /dev/ttyACM0
groups | tr ' ' '\n' | grep dialout || sudo usermod -aG dialout "$USER"
# log out / back in after adding to dialout
```

## Data capture workflow

Capture is **belt-fed**: load one denomination, run the belt, and let coins
pass under the camera. `capture.py` crops to the belt ROI, runs a coin-presence
gate so empty-belt frames are not saved, and tight-crops around each coin so it
fills the saved image (which is what the 224×224 whole-image classifier wants).

A **WS2812 ring light** (driven by the Pico on GP16) is turned on for every
capture and calibration session and off on exit, for even, consistent
illumination. It defaults to neutral white (`capture.light_rgb`) so copper and
silver render true; lower the values if frames are over-exposed, or pass
`--no-lights` to disable. Wiring: ring data → Pico GP16 (330–470 Ω series, 5 V
logic level-shift recommended), 5 V from a supply with the ground tied to the
Pico, ~1000 µF across the ring's 5 V/GND.

### Web UI (recommended on a headless Pi)

```bash
python -m coin_sorter.webcal        # then open http://<pi-ip>:8080/
```

A browser tool with a **live preview + coin-detection overlay** and sliders for
**focus** (lens position in dioptres), **white balance** (red/blue gains — fix
the magenta cast the LEDs cause), **ROI**, and the **gate thresholds**. You can
also **run belt-fed capture** from the page: set a label, Start, and watch the
saved count climb as coins go by. "Show config" emits a `config.local.yaml`
snippet so the tuned focus/WB/ROI/thresholds lock into the CLI tools.

The web UI **owns the camera while running** — stop it before using the CLI
`capture`/`--calibrate` commands below (only one process can open the camera).

### 1. Calibrate the ROI and thresholds (CLI, one-time)

```bash
python -m coin_sorter.capture --label _calib --calibrate
```

This grabs a frame, runs the detector, and writes `roi_snapshot.jpg` +
`overlay.jpg` (detected contour/bbox/centroid and the dedup band) to
`data/processed/_calib/`, logging the ROI rectangle, blob `area_frac`, and
`circularity`. Copy the snapshot to a workstation, then set the belt region and
tune the gate in **`config.local.yaml`** (gitignored, deep-merged over
`config.yaml`):

```yaml
camera:
  roi: [0.20, 0.15, 0.60, 0.70]   # x, y, w, h as fractions of the full frame
capture:
  min_area_frac: 0.02             # bracket the area_frac the calibrator logged
  max_area_frac: 0.60
  min_circularity: 0.65
  # invert: true                  # if coins are darker than the belt
  # method: absdiff               # if coin/belt brightness are too close to threshold
```

Re-run `--calibrate` with a coin in the ROI until the overlay shows exactly one
clean blob on the coin (and empty belt yields no detection). Then delete the
`_calib` snapshots before zipping.

### 2. Capture each class

```bash
python -m coin_sorter.capture --label penny   --count 250
python -m coin_sorter.capture --label nickel  --count 250
python -m coin_sorter.capture --label dime    --count 250
python -m coin_sorter.capture --label quarter --count 250
# 'reject' (foreign coins, debris, multi-coin clusters, empty belt) bypasses the
# single-coin gate automatically; feed junk and use --interval for cadence:
python -m coin_sorter.capture --label reject  --interval 0.4 --count 250
```

With the gate on (the default), **`--count` counts coins saved**, not frames
grabbed. By default one image is saved per coin as its centroid crosses the
middle of the ROI (`dedup_mode: centroid_band`). For more pose variety per coin
use `--dedup min_interval`; for every gated frame use `--dedup none`. Pass
`--no-gate` to fall back to the legacy "save the ROI-cropped frame every
`--interval`" behaviour.

Rough rule of thumb: 200–500 images per class is enough for a nano YOLO-cls to
converge. Aim for balanced counts (see "Class imbalance" below). Vary lighting
and orientation across the run, and spot-check `data/raw/<label>/` afterwards —
coins should be centred and filling the frame, with no empty-belt shots.

### Optional: drive the belt from the capture tool

```bash
python -m coin_sorter.capture --label penny --count 250 --drive-belt --belt-speed 800
```

This issues the firmware's non-blocking `RUN <hz>` command, which drives the
stepper from a hardware timer so the belt runs continuously while we capture
(the link stays responsive; the tool sends `STOP` on exit). Any Pico error is
logged and capture continues belt-less. Default is **off**. Keep the speed
gentle so coins don't blow past the dedup band between frames. The firmware
lives in `firmware/main.py`.

## Training workflow (Colab)

1. **Bundle the dataset on the Pi:**
   ```bash
   ./scripts/zip_dataset.sh
   ```
   Produces `./data.zip`.

2. **Upload to Drive** at `MyDrive/coin_sorter/data.zip`. The notebook reads
   this exact path.

3. **Open `notebooks/train_colab.ipynb` in Colab:**
   - Runtime → Change runtime type → T4 GPU
   - Runtime → Run all
   - The notebook mounts Drive, unzips, trains, prints top-1/top-5, writes a
     confusion matrix, exports ONNX, and copies the result to
     `MyDrive/coin_sorter/models/coin_classifier.onnx`.

4. **Pull the model back to the Pi:**
   ```bash
   # If you have Drive mounted on a workstation:
   ./scripts/fetch_model.sh user@workstation:/path/to/Drive/MyDrive/coin_sorter/models/coin_classifier.onnx

   # Or a local copy:
   ./scripts/fetch_model.sh ~/Downloads/coin_classifier.onnx
   ```
   The model lands at `models/coin_classifier.onnx` (the path in `config.yaml`).

## Deployment

```bash
source .venv/bin/activate
python -m coin_sorter.sorter
```

If `models/coin_classifier.onnx` is missing, the sorter logs a warning and
runs with a stub classifier that always returns `reject` with confidence 0.
This lets you verify the camera and Pico paths before any model exists.

## Configuration

Everything lives in `config.yaml`. Per-machine overrides go in
`config.local.yaml` (gitignored, deep-merged on top). The most useful knobs:

- `camera.width / height` — resolution. 1280×720 is plenty for the classifier
  and keeps frame rate up.
- `model.path` — where the ONNX model is loaded from.
- `model.input_size` — must match the `imgsz` you trained at.
- `classifier.confidence_threshold` — coins below this go to `reject`.
- `classifier.labels` — class order. **Must match alphabetical sort of the
  training class folders** (this is how Ultralytics assigns indices).
- `sorter.cooldown_ms` — guard against double-classifying the same coin.
- `camera.roi` — belt region as `[x, y, w, h]` fractions; capture crops to this.
- `camera.af_mode / lens_position` — Arducam 64MP focus (manual lock recommended
  for the fixed-distance belt; find the value in the web UI or cam_test/stream.py).
- `camera.awb / colour_gains` — white balance; lock it (manual `[red, blue]`) for
  consistent colour under the ring. Tune in the web UI (`coin_sorter.webcal`).
- `capture.*` — the belt-fed capture gate (segmentation method, area/circularity
  thresholds, dedup mode, crop padding). See "Data capture workflow" above and
  the inline comments in `config.yaml`.
- `capture.lights / light_rgb` — WS2812 ring light (Pico GP16); on for every
  session, neutral white by default.

## Troubleshooting

**`ModuleNotFoundError: picamera2`** — Your venv does not see system
site-packages. Recreate with `python -m venv --system-site-packages .venv`.

**`PermissionError: '/dev/ttyACM0'`** — You are not in the `dialout` group.
Run `sudo usermod -aG dialout $USER`, then log out and back in.

**`/dev/ttyACM0` missing** — The Pico is not enumerating. Check the USB cable
(many "charging" cables have no data lines), `dmesg | tail -20` for kernel
hints, or try a different USB port.

**`libcamera-hello` shows no cameras** — Confirm the CSI ribbon is seated
correctly, then ensure `dtoverlay=imx708` (Module 3) or the appropriate
overlay is enabled in `/boot/firmware/config.txt`. Reboot after changes.

**ONNX Runtime warns about provider** — The Pi build of `onnxruntime` from
PyPI ships with CPUExecutionProvider only. `XnnpackExecutionProvider` will
load only if you have an onnxruntime build with XNNPACK enabled; otherwise
CPU is fine — a YOLO-cls nano runs at >30 FPS at 224×224 on the Pi 5.

**Classifier always returns `reject` with confidence 0.0** — The stub is
active. Either the model file is missing, or the path in `config.yaml`
doesn't match where you placed it. Check `ls -lh models/`.

**Wrong class predicted consistently for one class** — Almost always a label-
order mismatch. The order of `classifier.labels` in `config.yaml` must match
`sorted(os.listdir(data_root))` from training time.

## Decisions worth flagging (see the Colab notebook for the full rationale)

- **Base model: YOLO11 nano-cls** by default. v11 trains a bit faster than v8
  for the same accuracy on small classification problems and the API is
  identical (`YOLO('yolo11n-cls.pt')`). Swap to `'yolov8n-cls.pt'` if you want
  to A/B; either is fine.
- **Image size: 224×224.** Coins-as-discs are low-frequency enough that this
  is plenty. If you later need to distinguish e.g. state quarter designs,
  bump to 320 or 384 (and update both `training.imgsz` and `model.input_size`).
- **Class imbalance.** Easiest fix is collecting balanced data — aim for the
  same image count per class. If that's impractical, oversample the minority
  classes (Ultralytics doesn't expose loss weighting cleanly for cls). The
  notebook prints a confusion matrix — watch for one row consistently mis-
  classified, that's the signal.
- **Augmentation.** Full 360° rotation is non-optional (coins land in any
  orientation). HSV jitter handles copper-vs-silver lighting drift. Avoid
  random crop / perspective — those distort the coin's circular outline,
  which is genuine signal.
- **ONNX export.** `dynamic=False` + fixed `imgsz` gives the smallest, fastest
  graph. Opset 17 is well-supported by onnxruntime 1.17+. If you ever change
  `imgsz`, re-export — input shape is baked in.
