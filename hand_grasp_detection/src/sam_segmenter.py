from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

Rect = tuple[int, int, int, int]

SAM_TYPE_MOBILE = "mobile_sam"
SAM_TYPE_SAM = "sam"


class SamSegmenter:
    """Bbox-prompt segmenter supporting both MobileSAM and SAM.

    sam_type: "mobile_sam" (default) or "sam"
    model_type: "vit_t" for MobileSAM, "vit_b"/"vit_l"/"vit_h" for SAM
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_t",
        device: str = "cpu",
        sam_type: str = SAM_TYPE_MOBILE,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise RuntimeError(f"SAM checkpoint not found: {checkpoint}")

        if sam_type == SAM_TYPE_MOBILE:
            try:
                from mobile_sam import SamPredictor, sam_model_registry
            except ImportError as exc:
                raise RuntimeError(
                    "mobile_sam is required: pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
                ) from exc
        elif sam_type == SAM_TYPE_SAM:
            try:
                from segment_anything import SamPredictor, sam_model_registry
            except ImportError as exc:
                raise RuntimeError(
                    "segment_anything is required: pip install git+https://github.com/facebookresearch/segment-anything.git"
                ) from exc
        else:
            raise ValueError(f"Unknown sam_type: {sam_type!r}. Choose 'mobile_sam' or 'sam'.")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def set_image(self, frame_bgr: np.ndarray) -> None:
        """Encode image once. Call before one or more predict_box() calls."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)

    def predict_box(self, bbox: Rect) -> Optional[np.ndarray]:
        """Return boolean mask for bbox. Must call set_image() first."""
        box = np.array(bbox, dtype=np.float32)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            return None
        return masks[0].astype(bool)

    def segment(self, frame_bgr: np.ndarray, bbox: Rect) -> Optional[np.ndarray]:
        """Convenience wrapper: set_image + predict_box in one call."""
        self.set_image(frame_bgr)
        return self.predict_box(bbox)
