"""ONNX Runtime inference for the coin classifier (Pi-side).

Loads a YOLO-cls ONNX export and exposes a tiny ``predict(frame)`` API. If the
model file is missing, falls back to a deterministic stub classifier so the
rest of the pipeline can be exercised end-to-end before any training has
happened.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger("coin_sorter.infer")


class CoinClassifier:
    """ONNX classifier wrapper. Stubs out gracefully if the model is missing.

    Parameters
    ----------
    model_path:
        Path to a YOLO-cls ONNX export. If the file does not exist, the
        classifier will run in stub mode and always return ``stub_label``.
    labels:
        Class labels, in the index order the model was trained with.
    input_size:
        Square input size the model was exported at (must match training).
    channel_order:
        ``"rgb"`` or ``"bgr"`` — YOLO exports want RGB; OpenCV gives BGR.
    mean / std:
        Per-channel normalization in 0..1 space. Ultralytics ONNX exports
        bake scaling in, so identity (mean=0, std=1) is correct for them.
    providers:
        ONNX Runtime execution providers. Defaults to XNNPACK then CPU.
    stub_label:
        Label returned in stub mode.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels: Sequence[str],
        input_size: int = 224,
        channel_order: str = "rgb",
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std: Sequence[float] = (1.0, 1.0, 1.0),
        providers: Sequence[str] | None = None,
        stub_label: str = "reject",
    ) -> None:
        self.labels = list(labels)
        self.input_size = int(input_size)
        self.channel_order = channel_order.lower()
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.stub_label = stub_label

        self._session = None
        self._input_name: str | None = None
        path = Path(model_path)
        if path.exists():
            self._load_session(path, providers)
        else:
            log.warning(
                "Model file %s not found — running in STUB mode. Predictions "
                "will always return %r with confidence 0.0.",
                path,
                self.stub_label,
            )

    def _load_session(self, path: Path, providers: Sequence[str] | None) -> None:
        import onnxruntime as ort

        if providers is None:
            available = set(ort.get_available_providers())
            providers = [p for p in ("XnnpackExecutionProvider", "CPUExecutionProvider") if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]
        log.info("Loading ONNX model %s with providers %s", path, providers)
        self._session = ort.InferenceSession(str(path), providers=list(providers))
        self._input_name = self._session.get_inputs()[0].name

    @property
    def is_stub(self) -> bool:
        """True if no real model is loaded."""
        return self._session is None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert an OpenCV BGR frame to model input tensor (NCHW float32)."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 frame, got shape {frame.shape}")
        img = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if self.channel_order == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        tensor = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        tensor = (tensor - self.mean) / self.std
        return tensor.astype(np.float32, copy=False)

    def predict(self, frame: np.ndarray) -> tuple[str, float]:
        """Classify a frame. Returns ``(label, confidence)``.

        Confidence is post-softmax in ``[0, 1]``.
        """
        if self._session is None:
            return self.stub_label, 0.0

        x = self.preprocess(frame)
        outputs = self._session.run(None, {self._input_name: x})
        logits = outputs[0]
        if logits.ndim > 1:
            logits = logits[0]
        probs = _as_probabilities(logits)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return label, conf

    @classmethod
    def from_config(cls, cfg: dict) -> "CoinClassifier":
        """Construct from the dict returned by :func:`coin_sorter.load_config`."""
        m = cfg["model"]
        return cls(
            model_path=m["path"],
            labels=cfg["classifier"]["labels"],
            input_size=int(m.get("input_size", 224)),
            channel_order=m.get("channel_order", "rgb"),
            mean=m.get("mean", (0.0, 0.0, 0.0)),
            std=m.get("std", (1.0, 1.0, 1.0)),
        )


def _as_probabilities(x: np.ndarray) -> np.ndarray:
    """Return a probability vector, applying softmax only if it is needed.

    Ultralytics YOLO-cls ONNX exports already end in a softmax, so the head
    emits probabilities, not logits. Running softmax over them again squashes
    the range: a fully confident 4-class prediction [1, 0, 0, 0] comes out as
    0.475, which sits below any sensible confidence threshold and sends every
    single coin to the `check` bin -- looking exactly like a bad model.

    Other export pipelines do emit raw logits, so detect rather than assume:
    a non-negative vector summing to 1 is already a distribution.
    """
    if x.size and x.min() >= 0.0 and abs(float(x.sum()) - 1.0) < 1e-3:
        return x
    return _softmax(x)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
