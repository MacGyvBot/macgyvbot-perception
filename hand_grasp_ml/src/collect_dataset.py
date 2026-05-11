from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

from feature_extractor import extract_features


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "hand_grasp_dataset.csv"
CAMERA_INDEX = 0
LABEL_KEYS = {
    ord("o"): "open",
    ord("g"): "grasp",
}
ACTIVE_LABELS = tuple(LABEL_KEYS.values())
WINDOW_NAME = "Hand Grasp Dataset Collector"


def ensure_data_dir() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_label_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    if not DATASET_PATH.exists():
        return counts

    with DATASET_PATH.open("r", newline="") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) == 64 and row[-1] in ACTIVE_LABELS:
                counts[row[-1]] += 1
    return counts


def append_sample(features: list[float], label: str) -> None:
    ensure_data_dir()
    with DATASET_PATH.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([*features, label])


def draw_status(frame, label_counts: Counter[str], hand_detected: bool) -> None:
    cv2.putText(
        frame,
        "o: open | g: grasp | ESC: exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"hand: {'detected' if hand_detected else 'not detected'}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if hand_detected else (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        (
            f"open: {label_counts['open']} | "
            f"grasp: {label_counts['grasp']}"
        ),
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def main() -> int:
    ensure_data_dir()
    label_counts = load_label_counts()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {CAMERA_INDEX}.")
        return 1

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    latest_features: list[float] | None = None

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Error: Failed to read frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                hand_detected = bool(results.multi_hand_landmarks)
                latest_features = None

                if hand_detected:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    latest_features = extract_features(hand_landmarks)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                draw_status(frame, label_counts, hand_detected)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in LABEL_KEYS:
                    if latest_features is None:
                        print("No hand detected. Move a hand into view before saving.")
                        continue
                    label = LABEL_KEYS[key]
                    append_sample(latest_features, label)
                    label_counts[label] += 1
                    print(
                        f"Saved label={label}. "
                        f"Total saved={sum(label_counts.values())}."
                    )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
