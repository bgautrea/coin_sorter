"""Smoke tests that exercise the package without touching real hardware."""

from __future__ import annotations

import numpy as np

from coin_sorter import load_config
from coin_sorter.infer import CoinClassifier


def test_config_loads() -> None:
    cfg = load_config()
    assert "camera" in cfg and "classifier" in cfg
    assert isinstance(cfg["classifier"]["labels"], list)


def test_stub_classifier_returns_reject() -> None:
    cfg = load_config()
    clf = CoinClassifier.from_config({**cfg, "model": {**cfg["model"], "path": "models/does-not-exist.onnx"}})
    assert clf.is_stub
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    label, conf = clf.predict(frame)
    assert label == "reject"
    assert conf == 0.0


def test_classifier_preprocess_shape() -> None:
    cfg = load_config()
    clf = CoinClassifier.from_config({**cfg, "model": {**cfg["model"], "path": "models/does-not-exist.onnx"}})
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    x = clf.preprocess(frame)
    assert x.shape == (1, 3, clf.input_size, clf.input_size)
    assert x.dtype == np.float32
