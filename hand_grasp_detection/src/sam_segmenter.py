from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

Rect = tuple[int, int, int, int]


class SamSegmenter:
    """SAM bbox-prompt segmenter for tool mask initialization."""

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda") -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("segment-anything is required for --sam-enabled") from exc

        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise RuntimeError(f"SAM checkpoint not found: {checkpoint}")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def segment(self, frame_bgr, bbox: Rect) -> Optional[np.ndarray]:
        """Return a boolean mask from a bbox prompt, or None."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)

        box = np.array(bbox, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None

        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)
