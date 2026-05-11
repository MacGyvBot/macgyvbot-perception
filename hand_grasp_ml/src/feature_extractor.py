from __future__ import annotations

from math import sqrt


EXPECTED_LANDMARK_COUNT = 21
FEATURE_COUNT = EXPECTED_LANDMARK_COUNT * 3
MIN_SCALE = 1e-6


def extract_features(hand_landmarks) -> list[float]:
    """
    Convert a single MediaPipe Hands landmark object into a 63-dimensional
    wrist-relative, scale-normalized feature vector.

    Args:
        hand_landmarks: Object with hand_landmarks.landmark[i].x/y/z values.

    Returns:
        A list of 63 float values ordered as x, y, z for each landmark.
    """
    landmarks = hand_landmarks.landmark
    if len(landmarks) != EXPECTED_LANDMARK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LANDMARK_COUNT} hand landmarks, "
            f"got {len(landmarks)}."
        )

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    scale = sqrt(
        (middle_mcp.x - wrist.x) ** 2
        + (middle_mcp.y - wrist.y) ** 2
        + (middle_mcp.z - wrist.z) ** 2
    )
    if scale < MIN_SCALE:
        scale = 1.0

    features: list[float] = []
    for point in landmarks:
        features.append(float((point.x - wrist.x) / scale))
        features.append(float((point.y - wrist.y) / scale))
        features.append(float((point.z - wrist.z) / scale))

    return features
