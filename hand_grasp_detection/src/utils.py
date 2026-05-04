from __future__ import annotations

from datetime import datetime
from math import hypot
from pathlib import Path
from typing import Optional, Tuple

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


def distance(p1: Point, p2: Point) -> float:
    """Return Euclidean distance between two image points."""
    return float(hypot(p1[0] - p2[0], p1[1] - p2[1]))


def point_in_rect(point: Point, rect: Rect, margin: int = 0) -> bool:
    """Return whether point is inside rect expanded by margin pixels."""
    x, y = point
    x1, y1, x2, y2 = rect
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def point_to_rect_distance(point: Point, rect: Rect) -> float:
    """Return shortest distance from a point to a rectangle; zero if inside."""
    x, y = point
    x1, y1, x2, y2 = rect
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(hypot(dx, dy))


def rect_from_points(points: list[Point]) -> Rect:
    """Return bounding rectangle for points."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def rect_intersection_area(rect_a: Rect, rect_b: Rect) -> float:
    """Return intersection area between two rectangles."""
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return float(width * height)


def rect_iou(rect_a: Rect, rect_b: Rect) -> float:
    """Return intersection-over-union for two rectangles."""
    intersection = rect_intersection_area(rect_a, rect_b)
    union = rect_area(rect_a) + rect_area(rect_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def rect_area(rect: Rect) -> float:
    """Return rectangle area."""
    x1, y1, x2, y2 = rect
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def expand_rect(rect: Rect, margin: int) -> Rect:
    """Return rectangle expanded by margin pixels."""
    x1, y1, x2, y2 = rect
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def median_depth_in_rect(depth_mm, rect: Rect, shrink_ratio: float = 0.2) -> Optional[float]:
    """Return median nonzero depth in a rectangle, in millimeters."""
    import numpy as np

    height, width = depth_mm.shape[:2]
    x1, y1, x2, y2 = _clip_rect(rect, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    shrink_x = int((x2 - x1) * shrink_ratio / 2)
    shrink_y = int((y2 - y1) * shrink_ratio / 2)
    x1, y1, x2, y2 = _clip_rect((x1 + shrink_x, y1 + shrink_y, x2 - shrink_x, y2 - shrink_y), width, height)
    values = depth_mm[y1:y2, x1:x2]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.median(values))


def median_depth_at_point(depth_mm, point: Point, radius: int = 3) -> Optional[float]:
    """Return median nonzero depth around an image point, in millimeters."""
    import numpy as np

    x, y = point
    rect = (x - radius, y - radius, x + radius + 1, y + radius + 1)
    height, width = depth_mm.shape[:2]
    x1, y1, x2, y2 = _clip_rect(rect, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    values = depth_mm[y1:y2, x1:x2]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.median(values))


def build_depth_grasp_info(
    hand_info: dict,
    tool_roi: Rect,
    depth_mm,
    depth_diff_threshold_mm: float,
    min_depth_contact_landmarks: int,
    roi_margin: int = 40,
) -> dict:
    """Build optional depth-contact metrics between hand landmarks and the tool ROI."""
    tool_depth = median_depth_in_rect(depth_mm, tool_roi)
    if tool_depth is None:
        return _empty_depth_info(tool_depth=None)

    expanded_tool_roi = expand_rect(tool_roi, roi_margin)
    valid_diffs = []
    depth_contact_count = 0

    for point in hand_info["landmarks"].values():
        if not point_in_rect(point, expanded_tool_roi):
            continue

        hand_depth = median_depth_at_point(depth_mm, point)
        if hand_depth is None:
            continue

        diff = abs(hand_depth - tool_depth)
        valid_diffs.append(diff)
        if diff <= depth_diff_threshold_mm:
            depth_contact_count += 1

    min_depth_diff = min(valid_diffs) if valid_diffs else None
    return {
        "depth_available": True,
        "tool_depth_mm": tool_depth,
        "min_hand_tool_depth_diff_mm": min_depth_diff,
        "depth_contact_count": depth_contact_count,
        "depth_grasp_confirmed": depth_contact_count >= min_depth_contact_landmarks,
    }


def _empty_depth_info(tool_depth: Optional[float]) -> dict:
    return {
        "depth_available": tool_depth is not None,
        "tool_depth_mm": tool_depth,
        "min_hand_tool_depth_diff_mm": None,
        "depth_contact_count": 0,
        "depth_grasp_confirmed": False,
    }


def _clip_rect(rect: Rect, width: int, height: int) -> Rect:
    x1, y1, x2, y2 = rect
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def draw_text(
    frame,
    text: str,
    position: Point,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """Draw outlined text for readability on live camera frames."""
    import cv2

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def save_screenshot(frame, directory: str = "logs") -> Path:
    """Save the current frame and return the created path."""
    import cv2

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"screenshot_{timestamp}.png"
    cv2.imwrite(str(path), frame)
    return path
