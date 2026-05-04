from __future__ import annotations

from typing import Optional, Tuple

from utils import (
    distance,
    expand_rect,
    point_in_rect,
    point_to_rect_distance,
    rect_area,
    rect_from_points,
    rect_intersection_area,
)

GRASP_HOLD_FRAMES = 15
PINCH_DISTANCE_THRESHOLD = 80
PALM_TO_TOOL_THRESHOLD = 80
ROI_MARGIN = 40
LANDMARK_TO_TOOL_THRESHOLD = 70
MIN_CONTACT_LANDMARKS = 2
HAND_TOOL_OVERLAP_THRESHOLD = 0.08
GRASP_SCORE_THRESHOLD = 3
DEPTH_GRASP_SCORE_BONUS = 2

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


class GraspDetector:
    """Heuristic detector for validating hand-tool grasp logic with a detected tool ROI."""

    def __init__(
        self,
        grasp_hold_frames: int = GRASP_HOLD_FRAMES,
        pinch_distance_threshold: float = PINCH_DISTANCE_THRESHOLD,
        palm_to_tool_threshold: float = PALM_TO_TOOL_THRESHOLD,
        roi_margin: int = ROI_MARGIN,
    ) -> None:
        self.grasp_hold_frames = grasp_hold_frames
        self.pinch_distance_threshold = pinch_distance_threshold
        self.palm_to_tool_threshold = palm_to_tool_threshold
        self.roi_margin = roi_margin
        self.grasp_counter = 0
        self.human_grasped_tool = False
        self.state = "NO_HAND"

    def update(self, hand_info: Optional[dict], tool_roi: Optional[Rect], depth_info: Optional[dict] = None) -> dict:
        """Update grasp state using hand landmarks and tool ROI."""
        if hand_info is None:
            self.reset()
            return self._result(None, None)

        if tool_roi is None:
            self.grasp_counter = 0
            self.human_grasped_tool = False
            self.state = "TOOL_NOT_DETECTED"
            return self._result(None, None)

        landmarks = hand_info["landmarks"]
        palm_center = hand_info["palm_center"]
        thumb_tip = hand_info["thumb_tip"]
        index_tip = hand_info["index_tip"]

        palm_to_tool_distance = point_to_rect_distance(palm_center, tool_roi)
        pinch_distance = distance(thumb_tip, index_tip)
        palm_near_roi = point_in_rect(palm_center, tool_roi, self.roi_margin)
        thumb_or_index_near_roi = self._thumb_or_index_near_tool(thumb_tip, index_tip, tool_roi)
        contact_count = self._count_contact_landmarks(landmarks, tool_roi)
        min_landmark_to_tool_distance = min(point_to_rect_distance(point, tool_roi) for point in landmarks.values())
        hand_tool_overlap_ratio = self._hand_tool_overlap_ratio(landmarks, tool_roi)

        self.state = "HAND_DETECTED"
        self.human_grasped_tool = False

        hand_near_tool = (
            palm_near_roi
            or palm_to_tool_distance < self.palm_to_tool_threshold
            or min_landmark_to_tool_distance < LANDMARK_TO_TOOL_THRESHOLD
            or contact_count >= MIN_CONTACT_LANDMARKS
            or hand_tool_overlap_ratio >= HAND_TOOL_OVERLAP_THRESHOLD
        )
        if hand_near_tool:
            self.state = "HAND_NEAR_TOOL"

        grasp_score = self._calculate_grasp_score(
            pinch_distance=pinch_distance,
            thumb_or_index_near_roi=thumb_or_index_near_roi,
            palm_near_roi=palm_near_roi,
            contact_count=contact_count,
            min_landmark_to_tool_distance=min_landmark_to_tool_distance,
            hand_tool_overlap_ratio=hand_tool_overlap_ratio,
            depth_info=depth_info,
        )

        # Use multiple contact signals instead of relying only on thumb-index pinch.
        grasp_candidate = hand_near_tool and grasp_score >= GRASP_SCORE_THRESHOLD

        if grasp_candidate:
            self.grasp_counter += 1
            self.state = "GRASP_CANDIDATE"
        else:
            self.grasp_counter = 0

        if self.grasp_counter >= self.grasp_hold_frames:
            self.state = "HUMAN_GRASPED_TOOL"
            self.human_grasped_tool = True

        return self._result(
            pinch_distance=pinch_distance,
            palm_to_tool_distance=palm_to_tool_distance,
            min_landmark_to_tool_distance=min_landmark_to_tool_distance,
            contact_count=contact_count,
            hand_tool_overlap_ratio=hand_tool_overlap_ratio,
            grasp_score=grasp_score,
            depth_info=depth_info,
        )

    def reset(self) -> None:
        self.grasp_counter = 0
        self.human_grasped_tool = False
        self.state = "NO_HAND"

    def _result(
        self,
        pinch_distance: Optional[float],
        palm_to_tool_distance: Optional[float],
        min_landmark_to_tool_distance: Optional[float] = None,
        contact_count: int = 0,
        hand_tool_overlap_ratio: float = 0.0,
        grasp_score: int = 0,
        depth_info: Optional[dict] = None,
    ) -> dict:
        depth_info = depth_info or {}
        return {
            "state": self.state,
            "grasp_counter": self.grasp_counter,
            "pinch_distance": pinch_distance,
            "palm_to_tool_distance": palm_to_tool_distance,
            "min_landmark_to_tool_distance": min_landmark_to_tool_distance,
            "contact_count": contact_count,
            "hand_tool_overlap_ratio": hand_tool_overlap_ratio,
            "grasp_score": grasp_score,
            "depth_available": depth_info.get("depth_available", False),
            "tool_depth_mm": depth_info.get("tool_depth_mm"),
            "min_hand_tool_depth_diff_mm": depth_info.get("min_hand_tool_depth_diff_mm"),
            "depth_contact_count": depth_info.get("depth_contact_count", 0),
            "depth_grasp_confirmed": depth_info.get("depth_grasp_confirmed", False),
            "human_grasped_tool": self.human_grasped_tool,
        }

    def _thumb_or_index_near_tool(self, thumb_tip: Point, index_tip: Point, tool_roi: Rect) -> bool:
        return point_in_rect(thumb_tip, tool_roi, self.roi_margin) or point_in_rect(
            index_tip,
            tool_roi,
            self.roi_margin,
        )

    def _count_contact_landmarks(self, landmarks: dict[int, Point], tool_roi: Rect) -> int:
        expanded_tool_roi = expand_rect(tool_roi, self.roi_margin)
        return sum(1 for point in landmarks.values() if point_in_rect(point, expanded_tool_roi))

    def _hand_tool_overlap_ratio(self, landmarks: dict[int, Point], tool_roi: Rect) -> float:
        hand_rect = rect_from_points(list(landmarks.values()))
        intersection = rect_intersection_area(hand_rect, expand_rect(tool_roi, self.roi_margin))
        hand_area = rect_area(hand_rect)
        if hand_area <= 0:
            return 0.0
        return intersection / hand_area

    def _calculate_grasp_score(
        self,
        pinch_distance: float,
        thumb_or_index_near_roi: bool,
        palm_near_roi: bool,
        contact_count: int,
        min_landmark_to_tool_distance: float,
        hand_tool_overlap_ratio: float,
        depth_info: Optional[dict],
    ) -> int:
        score = 0

        if contact_count >= MIN_CONTACT_LANDMARKS:
            score += 2
        elif min_landmark_to_tool_distance < LANDMARK_TO_TOOL_THRESHOLD:
            score += 1

        if thumb_or_index_near_roi:
            score += 1

        if pinch_distance < self.pinch_distance_threshold:
            score += 1

        if palm_near_roi or hand_tool_overlap_ratio >= HAND_TOOL_OVERLAP_THRESHOLD:
            score += 1

        if depth_info and depth_info.get("depth_grasp_confirmed", False):
            score += DEPTH_GRASP_SCORE_BONUS

        return score
