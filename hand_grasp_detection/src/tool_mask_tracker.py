from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

Rect = tuple[int, int, int, int]

MASK_DETECTED = "TOOL_MASK_DETECTED"
MASK_LOCKED = "TOOL_MASK_LOCKED"
MASK_OCCLUDED = "TOOL_MASK_OCCLUDED"
MASK_LOST = "TOOL_MASK_LOST"


@dataclass
class ToolMaskState:
    roi: Optional[Rect]
    mask: Optional[np.ndarray]
    state: str
    missing_frames: int
    visible_area_ratio: Optional[float]
    source: str


class ToolMaskTracker:
    """Maintain a locked tool mask so grasp checks survive short occlusions."""

    def __init__(
        self,
        max_missing_frames: int = 45,
        min_lock_area: int = 100,
        smoothing_alpha: float = 0.7,
    ) -> None:
        self.max_missing_frames = max_missing_frames
        self.min_lock_area = min_lock_area
        self.smoothing_alpha = smoothing_alpha
        self.locked_roi: Optional[Rect] = None
        self.locked_mask: Optional[np.ndarray] = None
        self.missing_frames = 0

    def reset(self) -> None:
        self.locked_roi = None
        self.locked_mask = None
        self.missing_frames = 0

    def update(
        self,
        detected_roi: Optional[Rect],
        detected_mask: Optional[np.ndarray],
        frame_shape: tuple[int, int],
    ) -> ToolMaskState:
        if detected_roi is not None:
            mask = detected_mask
            if mask is None:
                mask = self._rect_mask(detected_roi, frame_shape)

            if self._mask_area(mask) >= self.min_lock_area:
                self.locked_roi = self._smooth_roi(self.locked_roi, detected_roi)
                self.locked_mask = mask
                self.missing_frames = 0
                return ToolMaskState(
                    roi=self.locked_roi,
                    mask=self.locked_mask,
                    state=MASK_DETECTED,
                    missing_frames=self.missing_frames,
                    visible_area_ratio=1.0,
                    source="SAM" if detected_mask is not None else "YOLO_BBOX_MASK",
                )

        if self.locked_mask is not None and self.locked_roi is not None:
            self.missing_frames += 1
            state = (
                MASK_OCCLUDED
                if self.max_missing_frames < 0 or self.missing_frames <= self.max_missing_frames
                else MASK_LOST
            )
            if state == MASK_LOST:
                return ToolMaskState(
                    roi=None,
                    mask=None,
                    state=state,
                    missing_frames=self.missing_frames,
                    visible_area_ratio=0.0,
                    source="NONE",
                )

            return ToolMaskState(
                roi=self.locked_roi,
                mask=None,
                state=state,
                missing_frames=self.missing_frames,
                visible_area_ratio=0.0,
                source="LOCKED_MASK",
            )

        return ToolMaskState(
            roi=None,
            mask=None,
            state=MASK_LOST,
            missing_frames=self.missing_frames,
            visible_area_ratio=None,
            source="NONE",
        )

    def _smooth_roi(self, previous_roi: Optional[Rect], current_roi: Rect) -> Rect:
        if previous_roi is None:
            return current_roi

        alpha = self.smoothing_alpha
        return tuple(
            int(alpha * previous_value + (1.0 - alpha) * current_value)
            for previous_value, current_value in zip(previous_roi, current_roi)
        )

    def _mask_shape(self) -> tuple[int, int]:
        if self.locked_mask is not None:
            return self.locked_mask.shape[:2]
        return (0, 0)

    @staticmethod
    def _mask_area(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    @staticmethod
    def _rect_mask(rect: Rect, shape: tuple[int, int]) -> Optional[np.ndarray]:
        height, width = shape
        if height <= 0 or width <= 0:
            return None
        x1, y1, x2, y2 = rect
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
        return mask.astype(bool)
