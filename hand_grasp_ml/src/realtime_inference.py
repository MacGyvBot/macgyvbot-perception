from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

from feature_extractor import extract_features


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "hand_grasp_model.pkl"
CAMERA_INDEX = 0
WINDOW_NAME = "Hand Grasp Realtime Inference"
STABLE_WINDOW_SIZE = 10
STABLE_MIN_COUNT = 6
ACTIVE_MODEL_STATES = {"open", "grasp"}
STATE_COLORS = {
    "open": (0, 255, 255),
    "grasp": (0, 255, 0),
    "unstable": (0, 255, 255),
    "no_hand": (0, 255, 255),
}


def load_model():
    if not MODEL_PATH.exists():
        print(
            f"Error: Model not found at {MODEL_PATH}. "
            "Run python src/train_model.py first."
        )
        return None
    return joblib.load(MODEL_PATH)


def compute_stable_state(buffer: deque[str]) -> str:
    if not buffer:
        return "unstable"

    state, count = Counter(buffer).most_common(1)[0]
    if count >= STABLE_MIN_COUNT:
        return state
    return "unstable"


def predict_state(model, features: list[float]) -> tuple[str, float | None]:
    raw_state = str(model.predict([features])[0])
    if raw_state not in ACTIVE_MODEL_STATES:
        raw_state = "unstable"
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        confidence = float(max(probabilities))
    return raw_state, confidence


def draw_status(frame, raw_state: str, stable_state: str, confidence: float | None) -> None:
    raw_color = STATE_COLORS.get(raw_state, (255, 255, 255))
    stable_color = STATE_COLORS.get(stable_state, (255, 255, 255))
    cv2.putText(
        frame,
        f"Raw: {raw_state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        raw_color,
        2,
    )
    if confidence is not None:
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        stable_y = 100
    else:
        stable_y = 65

    cv2.putText(
        frame,
        f"Stable: {stable_state}",
        (10, stable_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        stable_color,
        2,
    )
    cv2.putText(
        frame,
        "Press ESC to exit",
        (10, stable_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def main() -> int:
    model = load_model()
    if model is None:
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {CAMERA_INDEX}.")
        return 1

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    stable_buffer: deque[str] = deque(maxlen=STABLE_WINDOW_SIZE)

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

                confidence = None
                if not results.multi_hand_landmarks:
                    raw_state = "no_hand"
                    stable_buffer.clear()
                    stable_state = "no_hand"
                else:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features = extract_features(hand_landmarks)
                    raw_state, confidence = predict_state(model, features)
                    stable_buffer.append(raw_state)
                    stable_state = compute_stable_state(stable_buffer)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                draw_status(frame, raw_state, stable_state, confidence)
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
