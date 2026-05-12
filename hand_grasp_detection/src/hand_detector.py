from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import distance, rect_from_points, rect_iou

Point = Tuple[int, int]

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
PINKY_MCP = 17

DUPLICATE_HAND_IOU_THRESHOLD = 0.35
DUPLICATE_PALM_DISTANCE_THRESHOLD = 90


class HandDetector:
    """Small wrapper around MediaPipe Hands for multi-hand landmark detection."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands_module = mp.solutions.hands
        self._hands = self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """Detect the first hand for backward compatibility, or None."""
        hands = self.detect_all(frame)
        if not hands:
            return None
        return hands[0]

    def detect_all(self, frame: np.ndarray) -> List[dict]:
        """Detect hand landmarks and return image-space coordinates for all hands."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            return []

        handedness_labels = self._get_handedness_labels(result)
        hands = []

        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks: Dict[int, Point] = {}

            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks[idx] = (x, y)

            palm_center = self._calculate_palm_center(landmarks)
            hand_rect = rect_from_points(list(landmarks.values()))

            hands.append(
                {
                    "hand_index": hand_index,
                    "handedness": handedness_labels[hand_index] if hand_index < len(handedness_labels) else "Unknown",
                    "landmarks": landmarks,
                    "hand_rect": hand_rect,
                    "palm_center": palm_center,
                    "thumb_tip": landmarks[THUMB_TIP],
                    "index_tip": landmarks[INDEX_TIP],
                    "middle_tip": landmarks[MIDDLE_TIP],
                    "ring_tip": landmarks[RING_TIP],
                    "pinky_tip": landmarks[PINKY_TIP],
                }
            )

        return self._deduplicate_hands(hands)

    def close(self) -> None:
        self._hands.close()

    @staticmethod
    def _calculate_palm_center(landmarks: Dict[int, Point]) -> Point:
        points = [landmarks[WRIST], landmarks[INDEX_MCP], landmarks[PINKY_MCP]]
        x = int(sum(point[0] for point in points) / len(points))
        y = int(sum(point[1] for point in points) / len(points))
        return (x, y)

    @staticmethod
    def _get_handedness_labels(result) -> List[str]:
        if not result.multi_handedness:
            return []

        labels = []
        for handedness in result.multi_handedness:
            if handedness.classification:
                labels.append(handedness.classification[0].label)
            else:
                labels.append("Unknown")
        return labels

    @staticmethod
    def _deduplicate_hands(hands: List[dict]) -> List[dict]:
        if len(hands) <= 1:
            return hands

        filtered_hands: List[dict] = []
        for hand in hands:
            is_duplicate = False
            for kept_hand in filtered_hands:
                same_region = rect_iou(hand["hand_rect"], kept_hand["hand_rect"]) >= DUPLICATE_HAND_IOU_THRESHOLD
                same_palm = (
                    distance(hand["palm_center"], kept_hand["palm_center"]) <= DUPLICATE_PALM_DISTANCE_THRESHOLD
                )
                if same_region or same_palm:
                    is_duplicate = True
                    break

            if not is_duplicate:
                hand["hand_index"] = len(filtered_hands)
                filtered_hands.append(hand)

        return filtered_hands
