from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_extractor import FEATURE_COUNT, extract_features  # noqa: E402


class DummyPoint:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class DummyHandLandmarks:
    def __init__(self, same_points: bool = False) -> None:
        if same_points:
            self.landmark = [DummyPoint(0.1, 0.2, 0.3) for _ in range(21)]
        else:
            self.landmark = [
                DummyPoint(i * 0.01, i * -0.02, i * 0.001) for i in range(21)
            ]


def test_extract_features_returns_63_values() -> None:
    features = extract_features(DummyHandLandmarks())

    assert len(features) == FEATURE_COUNT


def test_extract_features_returns_floats() -> None:
    features = extract_features(DummyHandLandmarks())

    assert all(isinstance(value, float) for value in features)


def test_extract_features_handles_near_zero_scale() -> None:
    features = extract_features(DummyHandLandmarks(same_points=True))

    assert len(features) == FEATURE_COUNT
    assert all(isinstance(value, float) for value in features)
