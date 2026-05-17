"""Reference training CLI — mirrors what `notebooks/train_colab.ipynb` does.

The notebook is the primary entry point; this module exists so the training
recipe stays under version control and reproducible from a workstation.

Pi note: this module imports `ultralytics` at runtime; it is in the optional
``[train]`` dependency group and is NOT installed on the Pi.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import configure_logging, load_config

log = logging.getLogger("coin_sorter.train")


def train(
    data_root: Path,
    base_model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    project: str,
    name: str,
    patience: int,
    resume: bool,
) -> Path:
    """Train a YOLO classifier and export to ONNX.

    Returns the path to the exported ``.onnx`` file.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "ultralytics is required for training. Install with: "
            "pip install '.[train]'"
        ) from e

    log.info("Loading base model: %s", base_model)
    model = YOLO(base_model)

    log.info("Starting training: epochs=%d batch=%d imgsz=%d data=%s", epochs, batch, imgsz, data_root)
    model.train(
        data=str(data_root),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        patience=patience,
        resume=resume,
        plots=True,
    )

    log.info("Exporting to ONNX (imgsz=%d, dynamic=False, opset=17)", imgsz)
    onnx_path_str = model.export(format="onnx", dynamic=False, imgsz=imgsz, opset=17)
    onnx_path = Path(onnx_path_str)
    log.info("Exported: %s", onnx_path)
    return onnx_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the coin classifier (reference CLI).")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    p.add_argument("--batch", type=int, default=None, help="Override training.batch.")
    p.add_argument("--imgsz", type=int, default=None, help="Override training.imgsz.")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    p.add_argument(
        "--data",
        default=None,
        help="Override dataset root. Should point at a folder-per-class layout.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = _parse_args(argv)
    cfg = load_config(args.config)
    tcfg = cfg["training"]

    data_root = Path(args.data) if args.data else Path(cfg["dataset"]["raw_dir"])
    onnx_path = train(
        data_root=data_root,
        base_model=tcfg["base_model"],
        epochs=args.epochs or int(tcfg["epochs"]),
        batch=args.batch or int(tcfg["batch"]),
        imgsz=args.imgsz or int(tcfg["imgsz"]),
        project=tcfg["project"],
        name=tcfg["name"],
        patience=int(tcfg.get("patience", 15)),
        resume=args.resume,
    )
    log.info("ONNX written to %s", onnx_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
