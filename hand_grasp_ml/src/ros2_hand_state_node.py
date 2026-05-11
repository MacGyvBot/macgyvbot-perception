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
NODE_NAME = "hand_grasp_state_node"
TOPIC_NAME = "/hand_grasp_state"
STABLE_WINDOW_SIZE = 10
STABLE_MIN_COUNT = 6
ACTIVE_MODEL_STATES = {"open", "grasp"}


def compute_stable_state(buffer: deque[str]) -> str:
    if not buffer:
        return "unstable"

    state, count = Counter(buffer).most_common(1)[0]
    if count >= STABLE_MIN_COUNT:
        return state
    return "unstable"


class HandGraspStateNode:
    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.String = String

        class _Node(Node):
            pass

        self.node = _Node(NODE_NAME)
        self.publisher = self.node.create_publisher(String, TOPIC_NAME, 10)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}.")

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run python src/train_model.py first."
            )
        self.model = joblib.load(MODEL_PATH)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.stable_buffer: deque[str] = deque(maxlen=STABLE_WINDOW_SIZE)
        self.timer = self.node.create_timer(0.01, self.process_frame)

    def process_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.node.get_logger().error("Failed to read frame from webcam.")
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self.stable_buffer.clear()
            stable_state = "no_hand"
        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            features = extract_features(hand_landmarks)
            raw_state = str(self.model.predict([features])[0])
            if raw_state not in ACTIVE_MODEL_STATES:
                raw_state = "unstable"
            self.stable_buffer.append(raw_state)
            stable_state = compute_stable_state(self.stable_buffer)

        msg = self.String()
        msg.data = stable_state
        self.publisher.publish(msg)

    def spin(self) -> None:
        self.rclpy.spin(self.node)

    def close(self) -> None:
        self.cap.release()
        self.hands.close()
        self.node.destroy_node()


def main() -> int:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy is not installed. Run this file in a ROS2 Humble environment.")
        return 1

    rclpy.init()
    node_wrapper: HandGraspStateNode | None = None
    try:
        node_wrapper = HandGraspStateNode()
        node_wrapper.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if node_wrapper is not None:
            node_wrapper.close()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
