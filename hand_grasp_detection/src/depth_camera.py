from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray


class RealSenseDepthCamera:
    """RealSense color/depth camera wrapper with depth aligned to color."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for --depth-source realsense") from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._align = rs.align(rs.stream.color)
        self._pipeline.start(self._config)

    def read(self) -> Optional[DepthFrame]:
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        return DepthFrame(color_bgr=color_bgr, depth_mm=depth_mm)

    def release(self) -> None:
        self._pipeline.stop()
