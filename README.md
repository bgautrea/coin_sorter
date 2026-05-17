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

Drop coins on the belt one denomination at a time, vary lighting and
position, and let `capture.py` accumulate images:

```bash
python -m coin_sorter.capture --label penny    --interval 0.5 --count 300
python -m coin_sorter.capture --label nickel   --interval 0.5 --count 300
python -m coin_sorter.capture --label dime     --interval 0.5 --count 300
python -m coin_sorter.capture --label quarter  --interval 0.5 --count 300
python -m coin_sorter.capture --label reject   --interval 0.5 --count 300  # foreign / junk
```

Rough rule of thumb: 200–500 images per class is enough for a nano YOLO-cls
to converge. Aim for balanced counts (see "Class imbalance" below).

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
