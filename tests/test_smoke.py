"""Smoke tests that exercise the package without touching real hardware."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from coin_sorter import load_config
from coin_sorter.capture import (
    CoinDetector,
    Deduper,
    Detection,
    crop_roi,
    resolve_roi,
    tight_square_crop,
)
from coin_sorter.infer import CoinClassifier


# --- helpers ---------------------------------------------------------------- #

_CAP_PARAMS = {
    "method": "brightness",
    "blur_ksize": 5,
    "min_area_frac": 0.01,
    "max_area_frac": 0.80,
    "min_circularity": 0.7,
    "bright_thresh": None,
    "dedup_mode": "centroid_band",
    "travel_axis": "y",
    "band_center_frac": 0.5,
    "band_halfwidth_frac": 0.15,
    "min_interval_s": 0.0,
}


def _belt_frame(w: int = 200, h: int = 200) -> np.ndarray:
    """A uniform mid-gray 'empty belt' BGR frame."""
    return np.full((h, w, 3), 90, dtype=np.uint8)


def _coin_frame(cx: int, cy: int, r: int = 30, w: int = 200, h: int = 200) -> np.ndarray:
    """A bright filled circle (coin) on the gray belt."""
    frame = _belt_frame(w, h)
    cv2.circle(frame, (cx, cy), r, (230, 230, 230), -1)
    return frame


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


# --- ROI -------------------------------------------------------------------- #


def test_resolve_roi_fractions_to_pixels() -> None:
    assert resolve_roi(None, 1280, 720) is None
    assert resolve_roi([0.25, 0.5, 0.5, 0.25], 1280, 720) == (320, 360, 640, 180)


def test_resolve_roi_clamps_to_frame() -> None:
    x, y, w, h = resolve_roi([0.9, 0.9, 0.5, 0.5], 100, 100)
    assert x + w <= 100 and y + h <= 100


def test_crop_roi_slices() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_roi(frame, (10, 20, 30, 40)).shape == (40, 30, 3)
    assert crop_roi(frame, None).shape == (100, 100, 3)


# --- detector --------------------------------------------------------------- #


def test_detector_finds_coin() -> None:
    det = CoinDetector(_CAP_PARAMS, 200, 200).detect(_coin_frame(100, 100, r=30))
    assert det.found
    assert det.circularity > 0.7
    cx, cy = det.centroid
    assert abs(cx - 100) <= 3 and abs(cy - 100) <= 3


def test_detector_ignores_empty_belt() -> None:
    assert not CoinDetector(_CAP_PARAMS, 200, 200).detect(_belt_frame()).found


def test_detector_area_filter_rejects_tiny_blob() -> None:
    params = {**_CAP_PARAMS, "min_area_frac": 0.2}  # circle r=10 is ~0.8% of area
    assert not CoinDetector(params, 200, 200).detect(_coin_frame(100, 100, r=10)).found


# --- deduper ---------------------------------------------------------------- #


def _det_at(cy: int) -> Detection:
    return Detection(found=True, centroid=(100, cy), bbox=(80, cy - 20, 40, 40), area=1200.0, circularity=0.9)


def test_deduper_fires_once_per_coin() -> None:
    d = Deduper("centroid_band", _CAP_PARAMS, 200, 200)  # band is y in [70, 130]
    assert not d.should_save(_det_at(40), 0.0)  # above band
    assert d.should_save(_det_at(100), 0.1)  # crosses into band -> save
    assert not d.should_save(_det_at(105), 0.2)  # same coin, already disarmed
    assert not d.should_save(Detection(found=False), 0.3)  # empty belt re-arms
    assert d.should_save(_det_at(100), 0.4)  # next coin saves


def test_deduper_min_interval_rate_caps() -> None:
    params = {**_CAP_PARAMS, "min_interval_s": 1.0}
    d = Deduper("min_interval", params, 200, 200)
    assert d.should_save(_det_at(100), 0.0)
    assert not d.should_save(_det_at(100), 0.5)  # within interval
    assert d.should_save(_det_at(100), 1.5)  # interval elapsed


# --- tight crop ------------------------------------------------------------- #


def test_tight_square_crop_is_square_centered() -> None:
    roi = _coin_frame(100, 100, r=30)
    det = CoinDetector(_CAP_PARAMS, 200, 200).detect(roi)
    crop = tight_square_crop(roi, det, pad_frac=0.25, square=True)
    assert crop is not None and crop.shape[0] == crop.shape[1]


def test_tight_square_crop_clamps_at_edge() -> None:
    roi = _coin_frame(10, 10, r=8)  # coin in the corner
    det = CoinDetector({**_CAP_PARAMS, "min_area_frac": 0.001}, 200, 200).detect(roi)
    crop = tight_square_crop(roi, det, pad_frac=0.5, square=True)
    assert crop is not None and crop.size > 0  # clamped, never out of bounds


# --- webcal pure functions -------------------------------------------------- #


def test_webcal_default_state_from_config() -> None:
    from coin_sorter.webcal import default_state

    s = default_state(load_config())
    assert s["focus_mode"] in ("continuous", "manual")
    assert 0.0 <= s["min_area"] <= 1.0 and "label" in s


def test_webcal_apply_setting_parses_and_flags_hardware() -> None:
    from coin_sorter.webcal import apply_setting, default_state

    s = default_state(load_config())
    # lens slider implies manual focus and flags a hardware change
    assert apply_setting(s, "lens", "4.5") is True
    assert s["lens"] == 4.5 and s["focus_mode"] == "manual"
    # red gain implies manual WB
    assert apply_setting(s, "red", "1.8") is True
    assert s["awb_mode"] == "manual"
    # ROI / thresholds are software-only (no hardware flag)
    assert apply_setting(s, "roi_w", "0.6") is False
    assert s["roi_w"] == 0.6
    # out-of-range clamps, garbage is ignored
    assert apply_setting(s, "lens", "999") is True and s["lens"] == 15.0
    apply_setting(s, "circ", "not_a_number")  # no raise


def test_webcal_config_snippet_round_trips_yaml() -> None:
    import yaml

    from coin_sorter.webcal import apply_setting, config_snippet, default_state

    s = default_state(load_config())
    apply_setting(s, "lens", "4.5")
    apply_setting(s, "red", "1.8")
    apply_setting(s, "blue", "1.5")
    apply_setting(s, "min_area", "0.03")
    assert apply_setting(s, "exp", "1500") is True and s["exp_mode"] == "manual"
    apply_setting(s, "gain", "3.0")
    parsed = yaml.safe_load(config_snippet(s))
    assert parsed["camera"]["af_mode"] == "manual"
    assert parsed["camera"]["lens_position"] == 4.5
    assert parsed["camera"]["awb"] == "manual"
    assert parsed["camera"]["colour_gains"] == [1.8, 1.5]
    assert parsed["camera"]["exposure_us"] == 1500
    assert parsed["camera"]["analogue_gain"] == 3.0
    assert parsed["capture"]["min_area_frac"] == 0.03


# --------------------------------------------------------------------------- #
# DivertQueue — the camera decides ~30 s upstream of the diverter, so the
# scheduling has to be right for a dozen coins in flight at once. None of this
# is observable on the rig without mis-sorting real coins, so test it here.
# --------------------------------------------------------------------------- #
def _queue(delay=30.0, hold=0.5):
    from coin_sorter.sorter import DivertQueue

    return DivertQueue(delay_s=delay, hold_s=hold)


def test_divert_is_not_due_until_the_coin_arrives():
    q = _queue()
    q.schedule("keep", "penny", now=100.0)
    assert q.due(now=100.0) == []
    assert q.due(now=129.9) == []
    assert len(q) == 1
    (bin_, label, late, missed), = q.due(now=130.0)
    assert (bin_, label, missed) == ("keep", "penny", False)
    assert late == 0.0
    assert len(q) == 0


def test_many_coins_in_flight_keep_their_own_bins():
    """The failure this class exists to prevent: 15 coins between the camera
    and the nose, each landing in whichever bin the newest coin asked for."""
    q = _queue()
    bins = ["keep", "common", "check"] * 5
    for i, b in enumerate(bins):
        q.schedule(b, f"coin{i}", now=100.0 + 2.0 * i)
    assert len(q) == 15
    assert q.due(now=125.0) == []          # none has arrived yet
    # Each becomes due 30 s after its own detection, in detection order.
    seen = []
    for i in range(len(bins)):
        for bin_, label, _late, missed in q.due(now=130.0 + 2.0 * i):
            assert not missed
            seen.append((bin_, label))
    assert seen == [(b, f"coin{i}") for i, b in enumerate(bins)]


def test_coin_past_its_window_is_reported_missed_not_diverted_late():
    """A late divert is worse than none: the coin has already tipped off, so
    aiming for it would mis-sort whichever coin is under the nose now."""
    q = _queue(hold=0.5)
    q.schedule("keep", "penny", now=100.0)
    (bin_, label, late, missed), = q.due(now=131.0)   # 1.0 s past, hold 0.5 s
    assert missed is True
    assert late == pytest.approx(1.0)


def test_within_hold_window_still_diverts():
    q = _queue(hold=0.5)
    q.schedule("keep", "penny", now=100.0)
    (_bin, _label, late, missed), = q.due(now=130.4)
    assert missed is False
    assert late == pytest.approx(0.4)
