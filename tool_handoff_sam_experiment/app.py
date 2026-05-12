from __future__ import annotations

import csv
import sys
import time
from collections import Counter, deque
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parent
SHARED_SRC = REPO_ROOT / "hand_grasp_detection" / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from grasp_detector import GraspDetector  # noqa: E402
from tool_detector import ToolDetection  # noqa: E402
from utils import build_mask_grasp_info, distance, draw_text, point_to_rect_distance, rect_from_points, rect_iou, save_screenshot  # noqa: E402

Rect = tuple[int, int, int, int]

WINDOW_NAME = "SAM Tool Handoff Experiment"
LOG_DIR = EXPERIMENT_ROOT / "logs"
ARTIFACT_DIR = EXPERIMENT_ROOT / "vlm_segments"
DEFAULT_YOLO_MODEL = str(
    EXPERIMENT_ROOT / "merge.pt"
    if (EXPERIMENT_ROOT / "merge.pt").exists()
    else EXPERIMENT_ROOT / "yolo_v11_merge.pt"
)
DEFAULT_GRASP_MODEL = str(EXPERIMENT_ROOT / "hand_grasp_model.pkl")
DEFAULT_SAM_CHECKPOINT = str(REPO_ROOT / "hand_grasp_detection" / "models" / "mobile_sam.pt")
DEFAULT_TOOL_CLASSES = "drill,hammer,pliers,screwdriver,wrench,tape-measure"
ACTIVE_GRASP_STATES = {"grasp"}
ACTIVE_MODEL_STATES = {"open", "grasp"}
EXPECTED_LANDMARK_COUNT = 21
FEATURE_COUNT = EXPECTED_LANDMARK_COUNT * 3
MIN_FEATURE_SCALE = 1e-6
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
DUPLICATE_HAND_IOU_THRESHOLD = 0.35
DUPLICATE_PALM_DISTANCE_THRESHOLD = 90


class BBoxPromptSegmenter:
    """SAM-family bbox prompt segmenter with MobileSAM as the default backend."""

    def __init__(
        self,
        backend: str,
        checkpoint_path: str,
        model_type: str,
        device: str,
        use_point_prompts: bool = True,
        clip_margin: int = 8,
        open_kernel: int = 3,
        close_kernel: int = 5,
        min_bbox_fill: float = 0.01,
        max_bbox_fill: float = 0.80,
        keep_largest_component: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise RuntimeError(f"SAM checkpoint not found: {checkpoint}")

        if backend == "mobile_sam":
            from mobile_sam import SamPredictor, sam_model_registry
        elif backend == "sam":
            from segment_anything import SamPredictor, sam_model_registry
        else:
            raise RuntimeError(f"unsupported SAM backend: {backend}")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.use_point_prompts = use_point_prompts
        self.clip_margin = clip_margin
        self.open_kernel = open_kernel
        self.close_kernel = close_kernel
        self.min_bbox_fill = min_bbox_fill
        self.max_bbox_fill = max_bbox_fill
        self.keep_largest_component = keep_largest_component

    def segment(
        self,
        frame_bgr: np.ndarray,
        bbox: Rect,
        prompt_points: Optional[list[tuple[int, int, int]]] = None,
    ) -> Optional[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)
        point_coords, point_labels = self._prompt_points(bbox, frame_bgr.shape[:2], prompt_points or [])
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox, dtype=np.float32),
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None
        best_idx = self._select_best_mask(masks, scores, bbox, frame_bgr.shape[:2])
        mask = masks[best_idx].astype(bool)
        return self._postprocess_mask(mask, bbox, frame_bgr.shape[:2])

    def _prompt_points(
        self,
        bbox: Rect,
        frame_shape: tuple[int, int],
        manual_points: list[tuple[int, int, int]],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.use_point_prompts and not manual_points:
            return None, None

        height, width = frame_shape
        x1, y1, x2, y2 = _clip_rect(bbox, width, height)
        if x2 <= x1 or y2 <= y1:
            return None, None

        points = []
        labels = []
        if self.use_point_prompts:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            inset_x = max(2, int((x2 - x1) * 0.12))
            inset_y = max(2, int((y2 - y1) * 0.12))
            points.extend(
                [
                    (cx, cy),
                    (x1 + inset_x, y1 + inset_y),
                    (x2 - inset_x, y1 + inset_y),
                    (x1 + inset_x, y2 - inset_y),
                    (x2 - inset_x, y2 - inset_y),
                ]
            )
            labels.extend([1, 0, 0, 0, 0])

        for x, y, label in manual_points:
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
                labels.append(label)

        if not points:
            return None, None
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _select_best_mask(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        bbox: Rect,
        frame_shape: tuple[int, int],
    ) -> int:
        bbox_mask = _rect_mask(_expand_and_clip_rect(bbox, 0, frame_shape), frame_shape)
        bbox_area = max(1, int(np.count_nonzero(bbox_mask)))
        best_idx = 0
        best_score = float("-inf")

        for idx, raw_mask in enumerate(masks):
            mask = raw_mask.astype(bool)
            area = int(np.count_nonzero(mask))
            if area == 0:
                continue
            inside_area = int(np.count_nonzero(mask & bbox_mask))
            outside_ratio = float((area - inside_area) / area)
            bbox_fill = float(inside_area / bbox_area)
            fill_penalty = 0.0
            if bbox_fill < self.min_bbox_fill:
                fill_penalty += self.min_bbox_fill - bbox_fill
            if bbox_fill > self.max_bbox_fill:
                fill_penalty += bbox_fill - self.max_bbox_fill

            combined_score = float(scores[idx]) - 0.70 * outside_ratio - 0.80 * fill_penalty
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        return best_idx

    def _postprocess_mask(self, mask: np.ndarray, bbox: Rect, frame_shape: tuple[int, int]) -> np.ndarray:
        clip_rect = _expand_and_clip_rect(bbox, self.clip_margin, frame_shape)
        clipped = mask & _rect_mask(clip_rect, frame_shape)
        processed = clipped.astype(np.uint8)

        if self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), dtype=np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        if self.open_kernel > 1:
            kernel = np.ones((self.open_kernel, self.open_kernel), dtype=np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        if self.keep_largest_component:
            processed = _largest_connected_component(processed)

        return processed.astype(bool)


@dataclass
class LockedToolMask:
    roi: Rect
    mask: np.ndarray
    source: str
    locked_at: str


@dataclass
class CandidateMask:
    roi: Rect
    mask: Optional[np.ndarray]
    label: str
    confidence: float
    source: str
    prompt_revision: int = 0


class MultiToolDetector:
    """YOLO wrapper that returns all matching tool detections plus an active one."""

    def __init__(
        self,
        model_path: str,
        target_classes: list[str],
        confidence_threshold: float,
        image_size: int,
        device: str,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.target_classes = {name.strip().lower() for name in target_classes if name.strip()}
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.model = YOLO(model_path)

    def detect_all(self, frame: np.ndarray) -> list[ToolDetection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections: list[ToolDetection] = []
        names = result.names
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id)).lower()
            if label not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                ToolDetection(
                    roi=(int(x1), int(y1), int(x2), int(y2)),
                    label=label,
                    confidence=confidence,
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections


@dataclass
class MLGraspResult:
    raw_state: str
    stable_state: str
    confidence: Optional[float]
    is_grasp: bool


class MLHandGraspClassifier:
    def __init__(self, model_path: str, stable_window_size: int, stable_min_count: int) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load the .pkl grasp model") from exc

        path = Path(model_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"grasp model not found: {path}")

        self.model = joblib.load(path)
        self.buffer: deque[str] = deque(maxlen=stable_window_size)
        self.stable_min_count = stable_min_count
        self.path = path

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, hand_info: Optional[dict]) -> MLGraspResult:
        if hand_info is None:
            self.reset()
            return MLGraspResult(raw_state="no_hand", stable_state="no_hand", confidence=None, is_grasp=False)

        features = _extract_ml_features(hand_info)
        raw_state = str(self.model.predict([features])[0])
        if raw_state not in ACTIVE_MODEL_STATES:
            raw_state = "unstable"

        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([features])[0]
            confidence = float(max(probabilities))

        self.buffer.append(raw_state)
        stable_state = _compute_stable_state(self.buffer, self.stable_min_count)
        is_grasp = raw_state in ACTIVE_GRASP_STATES and stable_state in ACTIVE_GRASP_STATES
        return MLGraspResult(
            raw_state=raw_state,
            stable_state=stable_state,
            confidence=confidence,
            is_grasp=is_grasp,
        )


class ExperimentHandDetector:
    """MediaPipe hand detector that keeps both pixel landmarks and ML features."""

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

    def detect_all(self, frame: np.ndarray) -> list[dict]:
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []

        handedness_labels = self._get_handedness_labels(result)
        hands = []
        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks: dict[int, tuple[int, int]] = {}
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmarks[idx] = (int(landmark.x * width), int(landmark.y * height))

            palm_center = self._calculate_palm_center(landmarks)
            hands.append(
                {
                    "hand_index": hand_index,
                    "handedness": handedness_labels[hand_index] if hand_index < len(handedness_labels) else "Unknown",
                    "landmarks": landmarks,
                    "hand_rect": rect_from_points(list(landmarks.values())),
                    "palm_center": palm_center,
                    "thumb_tip": landmarks[THUMB_TIP],
                    "index_tip": landmarks[INDEX_TIP],
                    "middle_tip": landmarks[MIDDLE_TIP],
                    "ring_tip": landmarks[RING_TIP],
                    "pinky_tip": landmarks[PINKY_TIP],
                    "ml_features": _extract_mediapipe_features(hand_landmarks),
                }
            )

        return self._deduplicate_hands(hands)

    def close(self) -> None:
        self._hands.close()

    @staticmethod
    def _calculate_palm_center(landmarks: dict[int, tuple[int, int]]) -> tuple[int, int]:
        points = [landmarks[WRIST], landmarks[INDEX_MCP], landmarks[PINKY_MCP]]
        return (int(sum(point[0] for point in points) / len(points)), int(sum(point[1] for point in points) / len(points)))

    @staticmethod
    def _get_handedness_labels(result) -> list[str]:
        if not result.multi_handedness:
            return []
        labels = []
        for handedness in result.multi_handedness:
            labels.append(handedness.classification[0].label if handedness.classification else "Unknown")
        return labels

    @staticmethod
    def _deduplicate_hands(hands: list[dict]) -> list[dict]:
        filtered_hands: list[dict] = []
        for hand in hands:
            is_duplicate = False
            for kept_hand in filtered_hands:
                same_region = rect_iou(hand["hand_rect"], kept_hand["hand_rect"]) >= DUPLICATE_HAND_IOU_THRESHOLD
                same_palm = distance(hand["palm_center"], kept_hand["palm_center"]) <= DUPLICATE_PALM_DISTANCE_THRESHOLD
                if same_region or same_palm:
                    is_duplicate = True
                    break
            if not is_duplicate:
                hand["hand_index"] = len(filtered_hands)
                filtered_hands.append(hand)
        return filtered_hands


def main() -> int:
    args = _parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera index {args.camera_index}")
        return 1
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    hand_detector = ExperimentHandDetector(max_num_hands=args.max_hands)
    contact_detector = GraspDetector(grasp_hold_frames=args.grasp_hold_frames)
    try:
        grasp_classifier = _create_grasp_classifier(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        hand_detector.close()
        cap.release()
        return 1
    tool_detector = _create_tool_detector(args)
    sam_segmenter = _create_sam_segmenter(args)
    csv_file, csv_writer = _open_csv_logger()

    locked_tool: Optional[LockedToolMask] = None
    manual_roi: Optional[Rect] = None
    latest_candidates: list[CandidateMask] = []
    latest_detection: Optional[ToolDetection] = None
    prompt_points: list[tuple[int, int, int]] = []
    prompt_revision = {"value": 0}
    frame_index = 0
    previous_frame_time = time.perf_counter()
    last_state_print = 0.0
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse_prompt, {"points": prompt_points, "revision": prompt_revision})

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: failed to read webcam frame")
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            frame_index += 1

            detections = _detect_tools(args, tool_detector, frame, manual_roi)
            detection = _select_active_detection(detections, latest_detection)
            if detection is not None:
                latest_detection = detection

            candidate: Optional[CandidateMask] = None
            if locked_tool is None:
                latest_candidates = _update_candidates(
                    args,
                    sam_segmenter,
                    frame,
                    detections,
                    latest_candidates,
                    frame_index,
                    prompt_points,
                    prompt_revision["value"],
                )
                candidate = _select_active_candidate(latest_candidates, detection)
                tool_roi = candidate.roi if candidate else None
                tool_mask = candidate.mask if candidate else None
                tool_source = _tool_source_from_detection(detection)
                mask_source = candidate.source if candidate else "NONE"
                phase = "PRE_LOCK"
            else:
                tool_roi = locked_tool.roi
                tool_mask = locked_tool.mask
                tool_source = _tool_source_from_detection(latest_detection)
                mask_source = locked_tool.source
                phase = "LOCKED"

            hand_infos = hand_detector.detect_all(frame)
            active_hand = _select_active_hand(hand_infos, tool_roi)
            hand_for_state = active_hand if active_hand is not None else (hand_infos[0] if hand_infos else None)
            mask_info = _build_mask_info(args, hand_for_state, tool_mask)
            contact_result = contact_detector.update(hand_for_state, tool_roi, None, mask_info)
            ml_result = grasp_classifier.update(hand_for_state)
            result = _compose_handoff_result(args, contact_result, ml_result)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_frame_time, 1e-6)
            previous_frame_time = now

            _draw_overlay(
                display=display,
                hand_infos=hand_infos,
                active_hand=active_hand,
                detection=detection,
                detections=detections,
                candidates=latest_candidates,
                active_candidate=candidate if locked_tool is None else None,
                locked_tool=locked_tool,
                result=result,
                phase=phase,
                tool_source=tool_source,
                mask_source=mask_source,
                fps=fps,
                prompt_points=prompt_points,
            )
            cv2.imshow(WINDOW_NAME, display)
            _write_csv_row(
                csv_writer,
                phase,
                tool_source,
                mask_source,
                latest_detection,
                candidate if locked_tool is None else None,
                locked_tool,
                active_hand,
                result,
            )

            if now - last_state_print >= args.print_interval:
                _print_state(phase, tool_source, mask_source, result)
                last_state_print = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                candidate = _select_active_candidate(latest_candidates, latest_detection)
                locked_tool = _lock_current_mask(args, frame, candidate, latest_detection)
                if locked_tool is not None:
                    contact_detector.reset()
                    grasp_classifier.reset()
                    print(f"INFO: locked tool mask roi={locked_tool.roi} source={locked_tool.source}")
            if key == ord("m"):
                selected = _select_manual_roi(frame)
                if selected is not None:
                    manual_roi = selected
                    latest_candidates = []
                    locked_tool = None
                    prompt_points.clear()
                    prompt_revision["value"] += 1
                    contact_detector.reset()
                    grasp_classifier.reset()
                    print(f"INFO: manual ROI set to {manual_roi}")
            if key == ord("v"):
                candidate = _select_active_candidate(latest_candidates, latest_detection)
                _save_vlm_artifacts(frame, candidate, locked_tool)
            if key == ord("s"):
                path = save_screenshot(display, str(LOG_DIR))
                print(f"INFO: screenshot saved to {path}")
            if key == ord("r"):
                manual_roi = None
                latest_candidates = []
                locked_tool = None
                prompt_points.clear()
                prompt_revision["value"] += 1
                contact_detector.reset()
                grasp_classifier.reset()
                print("INFO: reset all experiment state")
            if key == ord("c"):
                locked_tool = None
                prompt_points.clear()
                prompt_revision["value"] += 1
                contact_detector.reset()
                grasp_classifier.reset()
                print("INFO: cleared locked mask")
            if key == ord("p"):
                latest_candidates = []
                prompt_points.clear()
                prompt_revision["value"] += 1
                print("INFO: cleared SAM prompt points")
    finally:
        hand_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

    return 0


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Webcam SAM mask lock experiment for tool handoff.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--detector-source", choices=("yolo", "manual", "none"), default="yolo")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-device", default="cuda")
    parser.add_argument("--yolo-conf", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--tool-classes", default=DEFAULT_TOOL_CLASSES)
    parser.add_argument(
        "--requested-tool",
        default="",
        help="Single requested tool class. Overrides --tool-classes when set.",
    )
    parser.add_argument("--sam-enabled", action="store_true")
    parser.add_argument("--sam-backend", choices=("mobile_sam", "sam"), default="mobile_sam")
    parser.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default="vit_t")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--sam-interval", type=int, default=10)
    parser.add_argument("--max-sam-candidates", type=int, default=5)
    parser.add_argument("--no-sam-point-prompts", action="store_true")
    parser.add_argument("--sam-clip-margin", type=int, default=8)
    parser.add_argument("--sam-open-kernel", type=int, default=3)
    parser.add_argument("--sam-close-kernel", type=int, default=5)
    parser.add_argument("--sam-min-bbox-fill", type=float, default=0.01)
    parser.add_argument("--sam-max-bbox-fill", type=float, default=0.80)
    parser.add_argument("--no-sam-largest-component", action="store_true")
    parser.add_argument("--mask-contact-radius", type=int, default=6)
    parser.add_argument("--mask-min-contact-landmarks", type=int, default=2)
    parser.add_argument("--mask-min-lock-area", type=int, default=100)
    parser.add_argument("--allow-bbox-lock", action="store_true")
    parser.add_argument("--grasp-hold-frames", type=int, default=15)
    parser.add_argument("--grasp-model", default=DEFAULT_GRASP_MODEL)
    parser.add_argument("--grasp-stable-window", type=int, default=10)
    parser.add_argument("--grasp-stable-min-count", type=int, default=6)
    parser.add_argument(
        "--ml-grasp-mode",
        choices=("dominant", "require-contact"),
        default="dominant",
        help="dominant trusts stable ML grasp; require-contact also requires mask contact.",
    )
    parser.add_argument(
        "--ml-min-confidence",
        type=float,
        default=0.60,
        help="Minimum ML confidence for dominant grasp confirmation when predict_proba is available.",
    )
    parser.add_argument(
        "--contact-bonus-threshold",
        type=int,
        default=1,
        help="Mask contact count treated as supporting evidence, not a hard gate, in dominant mode.",
    )
    parser.add_argument(
        "--mask-proximity-threshold",
        type=float,
        default=45.0,
        help="Dominant mode still requires the hand to be within this pixel distance from the tool ROI/mask.",
    )
    parser.add_argument("--print-interval", type=float, default=1.0)
    return parser.parse_args()


def _create_grasp_classifier(args: Namespace) -> MLHandGraspClassifier:
    try:
        classifier = MLHandGraspClassifier(
            model_path=args.grasp_model,
            stable_window_size=args.grasp_stable_window,
            stable_min_count=args.grasp_stable_min_count,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to initialize ML grasp classifier: {exc}") from exc
    print(f"INFO: ML grasp classifier ready. model={classifier.path}")
    return classifier


def _create_tool_detector(args: Namespace) -> Optional[MultiToolDetector]:
    if args.detector_source != "yolo":
        print(f"INFO: YOLO disabled. detector_source={args.detector_source}")
        return None
    target_classes = _target_classes_from_args(args)
    try:
        detector = MultiToolDetector(
            model_path=args.yolo_model,
            target_classes=target_classes,
            confidence_threshold=args.yolo_conf,
            image_size=args.yolo_imgsz,
            device=args.yolo_device,
        )
    except Exception as exc:
        print(f"WARNING: YOLO init failed: {exc}")
        return None
    print(f"INFO: YOLO ready. model={args.yolo_model}, classes={target_classes}, device={args.yolo_device}")
    return detector


def _target_classes_from_args(args: Namespace) -> list[str]:
    if args.requested_tool.strip():
        return [_normalize_class_name(args.requested_tool)]
    return [_normalize_class_name(name) for name in args.tool_classes.split(",") if name.strip()]


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _create_sam_segmenter(args: Namespace) -> Optional[BBoxPromptSegmenter]:
    if not args.sam_enabled:
        print("INFO: SAM disabled. Use --sam-enabled to preview and lock SAM masks.")
        return None
    try:
        segmenter = BBoxPromptSegmenter(
            backend=args.sam_backend,
            checkpoint_path=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.sam_device,
            use_point_prompts=not args.no_sam_point_prompts,
            clip_margin=args.sam_clip_margin,
            open_kernel=args.sam_open_kernel,
            close_kernel=args.sam_close_kernel,
            min_bbox_fill=args.sam_min_bbox_fill,
            max_bbox_fill=args.sam_max_bbox_fill,
            keep_largest_component=not args.no_sam_largest_component,
        )
    except Exception as exc:
        print(f"WARNING: SAM init failed: {exc}")
        return None
    print(
        "INFO: SAM ready. backend={backend}, checkpoint={checkpoint}, model_type={model_type}, "
        "device={device}, point_prompts={point_prompts}, clip_margin={clip_margin}".format(
            backend=args.sam_backend,
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.sam_device,
            point_prompts=not args.no_sam_point_prompts,
            clip_margin=args.sam_clip_margin,
        )
    )
    return segmenter


def _detect_tools(
    args: Namespace,
    tool_detector: Optional[MultiToolDetector],
    frame: np.ndarray,
    manual_roi: Optional[Rect],
) -> list[ToolDetection]:
    if args.detector_source == "manual":
        if manual_roi is None:
            return []
        return [ToolDetection(roi=manual_roi, label="manual", confidence=1.0)]
    if args.detector_source == "yolo" and tool_detector is not None:
        return tool_detector.detect_all(frame)
    return []


def _select_active_detection(
    detections: list[ToolDetection],
    previous_detection: Optional[ToolDetection],
) -> Optional[ToolDetection]:
    if not detections:
        return None
    if previous_detection is None:
        return detections[0]

    previous_center = _rect_center(previous_detection.roi)
    same_label = [detection for detection in detections if detection.label == previous_detection.label]
    candidates = same_label or detections
    return min(candidates, key=lambda detection: point_to_rect_distance(previous_center, detection.roi))


def _on_mouse_prompt(event, x, y, flags, param) -> None:
    if param is None:
        return
    points: list[tuple[int, int, int]] = param["points"]
    revision: dict[str, int] = param["revision"]
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y), 1))
        revision["value"] += 1
        print(f"INFO: added SAM foreground point ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append((int(x), int(y), 0))
        revision["value"] += 1
        print(f"INFO: added SAM background point ({x}, {y})")


def _tool_source_from_detection(detection: Optional[ToolDetection]) -> str:
    if detection is None:
        return "NONE"
    if detection.label == "manual":
        return "MANUAL"
    return "YOLO"


def _update_candidate(
    args: Namespace,
    sam_segmenter: Optional[BBoxPromptSegmenter],
    frame: np.ndarray,
    detection: Optional[ToolDetection],
    previous: Optional[CandidateMask],
    frame_index: int,
    prompt_points: list[tuple[int, int, int]],
    prompt_revision: int,
) -> Optional[CandidateMask]:
    if detection is None:
        return previous

    should_segment = (
        previous is None
        or previous.roi != detection.roi
        or previous.prompt_revision != prompt_revision
        or frame_index % max(1, args.sam_interval) == 0
    )
    if not should_segment:
        return CandidateMask(
            detection.roi,
            previous.mask,
            detection.label,
            detection.confidence,
            previous.source,
            previous.prompt_revision,
        )

    mask = None
    source = "YOLO_BBOX"
    if sam_segmenter is not None:
        try:
            mask = sam_segmenter.segment(frame, detection.roi, prompt_points)
            source = "SAM_REFINED" if mask is not None else "YOLO_BBOX"
        except Exception as exc:
            print(f"WARNING: SAM segmentation failed: {exc}")

    return CandidateMask(detection.roi, mask, detection.label, detection.confidence, source, prompt_revision)


def _update_candidates(
    args: Namespace,
    sam_segmenter: Optional[BBoxPromptSegmenter],
    frame: np.ndarray,
    detections: list[ToolDetection],
    previous_candidates: list[CandidateMask],
    frame_index: int,
    prompt_points: list[tuple[int, int, int]],
    prompt_revision: int,
) -> list[CandidateMask]:
    if not detections:
        return previous_candidates

    updated_candidates: list[CandidateMask] = []
    used_previous: set[int] = set()
    for detection in detections[: max(1, args.max_sam_candidates)]:
        previous_index, previous = _match_previous_candidate(detection, previous_candidates, used_previous)
        if previous_index is not None:
            used_previous.add(previous_index)
        candidate_prompt_points = _prompt_points_for_roi(prompt_points, detection.roi)
        updated = _update_candidate(
            args,
            sam_segmenter,
            frame,
            detection,
            previous,
            frame_index,
            candidate_prompt_points,
            prompt_revision,
        )
        if updated is not None:
            updated_candidates.append(updated)
    return updated_candidates


def _match_previous_candidate(
    detection: ToolDetection,
    previous_candidates: list[CandidateMask],
    used_previous: set[int],
) -> tuple[Optional[int], Optional[CandidateMask]]:
    best_index = None
    best_score = float("inf")
    detection_center = _rect_center(detection.roi)
    for index, candidate in enumerate(previous_candidates):
        if index in used_previous or candidate.label != detection.label:
            continue
        distance_to_previous = point_to_rect_distance(detection_center, candidate.roi)
        iou_bonus = rect_iou(detection.roi, candidate.roi) * 100.0
        score = distance_to_previous - iou_bonus
        if score < best_score:
            best_score = score
            best_index = index
    if best_index is None:
        return None, None
    return best_index, previous_candidates[best_index]


def _select_active_candidate(
    candidates: list[CandidateMask],
    active_detection: Optional[ToolDetection],
) -> Optional[CandidateMask]:
    if not candidates:
        return None
    if active_detection is None:
        return candidates[0]
    same_label = [candidate for candidate in candidates if candidate.label == active_detection.label]
    search_space = same_label or candidates
    active_center = _rect_center(active_detection.roi)
    return min(search_space, key=lambda candidate: point_to_rect_distance(active_center, candidate.roi))


def _prompt_points_for_roi(prompt_points: list[tuple[int, int, int]], roi: Rect) -> list[tuple[int, int, int]]:
    expanded = _expand_rect_unclipped(roi, 30)
    x1, y1, x2, y2 = expanded
    return [(x, y, label) for x, y, label in prompt_points if x1 <= x <= x2 and y1 <= y <= y2]


def _lock_current_mask(
    args: Namespace,
    frame: np.ndarray,
    candidate: Optional[CandidateMask],
    latest_detection: Optional[ToolDetection],
) -> Optional[LockedToolMask]:
    if candidate is not None and candidate.mask is not None:
        area = int(np.count_nonzero(candidate.mask))
        if area >= args.mask_min_lock_area:
            roi = _mask_to_rect(candidate.mask) or candidate.roi
            return LockedToolMask(roi=roi, mask=candidate.mask.copy(), source="SAM_LOCKED", locked_at=_timestamp())
        print(f"WARNING: current SAM mask area is too small for lock: area={area}")
        return None

    if args.allow_bbox_lock:
        roi = candidate.roi if candidate is not None else latest_detection.roi if latest_detection is not None else None
        if roi is not None:
            return LockedToolMask(roi=roi, mask=_rect_mask(roi, frame.shape[:2]), source="BBOX_LOCKED", locked_at=_timestamp())

    print("WARNING: no SAM mask to lock. Use --sam-enabled or --allow-bbox-lock.")
    return None


def _build_mask_info(args: Namespace, hand_info: Optional[dict], tool_mask: Optional[np.ndarray]) -> Optional[dict]:
    if hand_info is None or tool_mask is None:
        return None
    return build_mask_grasp_info(
        hand_info=hand_info,
        tool_mask=tool_mask,
        contact_radius=args.mask_contact_radius,
        min_mask_contact_landmarks=args.mask_min_contact_landmarks,
    )


def _compose_handoff_result(args: Namespace, contact_result: dict, ml_result: MLGraspResult) -> dict:
    result = dict(contact_result)
    contact_confirmed = bool(result.get("mask_grasp_confirmed", False))
    contact_count = int(result.get("mask_contact_count", 0))
    contact_support = contact_count >= args.contact_bonus_threshold
    proximity_value = result.get("min_landmark_to_tool_distance")
    proximity_ok = proximity_value is not None and float(proximity_value) <= args.mask_proximity_threshold
    mask_near_or_contact = bool(contact_support or proximity_ok or contact_confirmed)
    confidence_ok = ml_result.confidence is None or ml_result.confidence >= args.ml_min_confidence

    ml_required = bool(ml_result.is_grasp)
    if not ml_required:
        human_grasped_tool = False
    elif args.ml_grasp_mode == "require-contact":
        human_grasped_tool = bool(confidence_ok and contact_confirmed)
    else:
        human_grasped_tool = bool(confidence_ok and mask_near_or_contact)

    result["contact_state"] = contact_result["state"]
    result["ml_raw_state"] = ml_result.raw_state
    result["ml_stable_state"] = ml_result.stable_state
    result["ml_confidence"] = ml_result.confidence
    result["ml_grasp_confirmed"] = ml_result.is_grasp
    result["ml_required"] = ml_required
    result["ml_confidence_ok"] = confidence_ok
    result["contact_support"] = contact_support
    result["mask_proximity_ok"] = proximity_ok
    result["mask_near_or_contact"] = mask_near_or_contact
    result["decision_mode"] = args.ml_grasp_mode
    result["human_grasped_tool"] = human_grasped_tool

    if human_grasped_tool:
        result["state"] = "HUMAN_GRASPED_TOOL"
    elif not ml_required and ml_result.stable_state in ACTIVE_GRASP_STATES:
        result["state"] = "ML_GRASP_BUFFER_NOT_RAW"
    elif not ml_required and (contact_confirmed or contact_support):
        result["state"] = "CONTACT_WITHOUT_ML_GRASP"
    elif ml_result.is_grasp and not confidence_ok:
        result["state"] = "ML_GRASP_LOW_CONF"
    elif ml_result.is_grasp and not mask_near_or_contact:
        result["state"] = "ML_GRASP_NOT_NEAR_TOOL"
    elif ml_result.is_grasp and contact_support:
        result["state"] = "ML_GRASP_WITH_CONTACT_SUPPORT"
    elif ml_result.is_grasp:
        result["state"] = "ML_GRASP_NO_CONTACT_SUPPORT"
    elif contact_confirmed:
        result["state"] = "MASK_CONTACT_NO_ML_GRASP"

    return result


def _extract_ml_features(hand_info: dict) -> list[float]:
    if "ml_features" in hand_info:
        return list(hand_info["ml_features"])

    landmarks = hand_info["landmarks"]
    if len(landmarks) != EXPECTED_LANDMARK_COUNT:
        raise ValueError(f"Expected {EXPECTED_LANDMARK_COUNT} hand landmarks, got {len(landmarks)}.")

    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    scale = sqrt((middle_mcp[0] - wrist[0]) ** 2 + (middle_mcp[1] - wrist[1]) ** 2)
    if scale < MIN_FEATURE_SCALE:
        scale = 1.0

    features: list[float] = []
    for index in range(EXPECTED_LANDMARK_COUNT):
        point = landmarks[index]
        features.append(float((point[0] - wrist[0]) / scale))
        features.append(float((point[1] - wrist[1]) / scale))
        features.append(0.0)

    if len(features) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {len(features)}.")
    return features


def _extract_mediapipe_features(hand_landmarks) -> list[float]:
    landmarks = hand_landmarks.landmark
    if len(landmarks) != EXPECTED_LANDMARK_COUNT:
        raise ValueError(f"Expected {EXPECTED_LANDMARK_COUNT} hand landmarks, got {len(landmarks)}.")

    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    scale = sqrt(
        (middle_mcp.x - wrist.x) ** 2
        + (middle_mcp.y - wrist.y) ** 2
        + (middle_mcp.z - wrist.z) ** 2
    )
    if scale < MIN_FEATURE_SCALE:
        scale = 1.0

    features: list[float] = []
    for point in landmarks:
        features.append(float((point.x - wrist.x) / scale))
        features.append(float((point.y - wrist.y) / scale))
        features.append(float((point.z - wrist.z) / scale))
    return features


def _compute_stable_state(buffer: deque[str], stable_min_count: int) -> str:
    if not buffer:
        return "unstable"
    state, count = Counter(buffer).most_common(1)[0]
    if count >= stable_min_count:
        return state
    return "unstable"


def _select_active_hand(hand_infos: list[dict], tool_roi: Optional[Rect]) -> Optional[dict]:
    if not hand_infos or tool_roi is None:
        return None
    return min(hand_infos, key=lambda hand_info: point_to_rect_distance(hand_info["palm_center"], tool_roi))


def _select_manual_roi(frame: np.ndarray) -> Optional[Rect]:
    roi = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=True, fromCenter=False)
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        print("INFO: manual ROI selection cancelled")
        return None
    return (x, y, x + width, y + height)


def _draw_overlay(
    display: np.ndarray,
    hand_infos: list[dict],
    active_hand: Optional[dict],
    detection: Optional[ToolDetection],
    detections: list[ToolDetection],
    candidates: list[CandidateMask],
    active_candidate: Optional[CandidateMask],
    locked_tool: Optional[LockedToolMask],
    result: dict,
    phase: str,
    tool_source: str,
    mask_source: str,
    fps: float,
    prompt_points: list[tuple[int, int, int]],
) -> None:
    if locked_tool is None:
        _draw_candidate_masks(display, candidates, active_candidate)
    if locked_tool is not None:
        _apply_mask_overlay(display, locked_tool.mask, (0, 255, 0), alpha=0.35)

    _draw_tool_detections(display, detections, detection)
    _draw_prompt_points(display, prompt_points)

    active_index = active_hand["hand_index"] if active_hand else None
    for hand_info in hand_infos:
        is_active = hand_info["hand_index"] == active_index
        color = (0, 255, 255) if is_active else (180, 180, 180)
        for point in hand_info["landmarks"].values():
            cv2.circle(display, point, 3 if is_active else 2, color, -1)
        cv2.circle(display, hand_info["palm_center"], 6, color, -1)
        cv2.line(display, hand_info["thumb_tip"], hand_info["index_tip"], (255, 255, 0), 2 if is_active else 1)

    state_color = _state_color(result["state"])
    y = 28
    for text, color in [
        (f"phase: {phase}", (255, 255, 255)),
        (f"state: {result['state']}", state_color),
        (f"human_grasped_tool: {result['human_grasped_tool']}", state_color),
        (f"ml_grasp: {result['ml_stable_state']} raw={result['ml_raw_state']}", state_color),
        (f"ml_conf: {_format_optional_float(result['ml_confidence'])}", (255, 255, 255)),
        (f"decision: {result['decision_mode']}", (255, 255, 255)),
        (f"mask_near: {result['mask_near_or_contact']} contact={result['mask_contact_count']}", (255, 255, 255)),
        (f"min_tool_dist: {_format_optional_float(result['min_landmark_to_tool_distance'])}", (255, 255, 255)),
        (f"grasp_score: {result['grasp_score']}", (255, 255, 255)),
        (f"tool: {tool_source}", (255, 255, 255)),
        (f"mask: {mask_source}", (255, 255, 255)),
        (f"detections: {len(detections)}", (255, 255, 255)),
        (f"hands: {len(hand_infos)}", (255, 255, 255)),
        (f"fps: {fps:.1f}", (255, 255, 255)),
    ]:
        draw_text(display, text, (18, y), color, scale=0.58)
        y += 28

    draw_text(display, "Left fg | Right bg | P clear pts | L lock | M ROI | V save | R reset | C clear | Q quit", (18, display.shape[0] - 18), (230, 230, 230), scale=0.45)


def _draw_rect(frame: np.ndarray, roi: Rect, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_text(frame, label, (x1, max(22, y1 - 8)), color, scale=0.5, thickness=1)


def _draw_tool_detections(
    frame: np.ndarray,
    detections: list[ToolDetection],
    active_detection: Optional[ToolDetection],
) -> None:
    for detection in detections:
        if detection.label == "manual":
            continue
        is_active = active_detection is not None and detection.roi == active_detection.roi and detection.label == active_detection.label
        color = (255, 180, 0) if is_active else (180, 180, 180)
        label = f"{detection.label} {detection.confidence:.2f}"
        if is_active:
            label = f"* {label}"
        _draw_rect(frame, detection.roi, color, label)


def _draw_candidate_masks(
    frame: np.ndarray,
    candidates: list[CandidateMask],
    active_candidate: Optional[CandidateMask],
) -> None:
    for candidate in candidates:
        if candidate.mask is None:
            continue
        is_active = (
            active_candidate is not None
            and candidate.roi == active_candidate.roi
            and candidate.label == active_candidate.label
        )
        color = (0, 220, 255) if is_active else (120, 200, 120)
        alpha = 0.30 if is_active else 0.18
        _apply_mask_overlay(frame, candidate.mask, color, alpha=alpha)


def _draw_prompt_points(frame: np.ndarray, prompt_points: list[tuple[int, int, int]]) -> None:
    for x, y, label in prompt_points:
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    overlay = frame.copy()
    overlay[mask.astype(bool)] = color
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)


def _save_vlm_artifacts(
    frame: np.ndarray,
    candidate: Optional[CandidateMask],
    locked_tool: Optional[LockedToolMask],
) -> None:
    if locked_tool is not None:
        mask = locked_tool.mask
        roi = locked_tool.roi
        prefix = f"{_timestamp()}_locked"
    elif candidate is not None and candidate.mask is not None:
        mask = candidate.mask
        roi = _mask_to_rect(candidate.mask) or candidate.roi
        prefix = f"{_timestamp()}_candidate"
    else:
        print("WARNING: no SAM mask available for VLM artifact")
        return

    x1, y1, x2, y2 = _clip_rect(roi, frame.shape[1], frame.shape[0])
    masked = np.zeros_like(frame)
    masked[mask.astype(bool)] = frame[mask.astype(bool)]
    crop = frame[y1:y2, x1:x2]
    mask_crop = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    masked_crop = masked[y1:y2, x1:x2]

    cv2.imwrite(str(ARTIFACT_DIR / f"{prefix}_frame.jpg"), frame)
    cv2.imwrite(str(ARTIFACT_DIR / f"{prefix}_crop.jpg"), crop)
    cv2.imwrite(str(ARTIFACT_DIR / f"{prefix}_mask.png"), mask_crop)
    cv2.imwrite(str(ARTIFACT_DIR / f"{prefix}_masked_crop.png"), masked_crop)
    print(f"INFO: saved VLM artifacts under {ARTIFACT_DIR} prefix={prefix}")


def _open_csv_logger() -> tuple[object, csv.DictWriter]:
    path = LOG_DIR / f"handoff_sam_{_timestamp()}.csv"
    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp",
            "phase",
            "state",
            "human_grasped_tool",
            "contact_state",
            "ml_raw_state",
            "ml_stable_state",
            "ml_confidence",
            "ml_grasp_confirmed",
            "ml_required",
            "ml_confidence_ok",
            "contact_support",
            "mask_proximity_ok",
            "mask_near_or_contact",
            "decision_mode",
            "grasp_counter",
            "grasp_score",
            "mask_contact_count",
            "mask_grasp_confirmed",
            "tool_source",
            "mask_source",
            "tool_label",
            "tool_confidence",
            "candidate_roi",
            "locked_roi",
            "active_hand_index",
            "active_handedness",
        ],
    )
    writer.writeheader()
    print(f"INFO: CSV log file: {path}")
    return csv_file, writer


def _write_csv_row(
    writer: csv.DictWriter,
    phase: str,
    tool_source: str,
    mask_source: str,
    detection: Optional[ToolDetection],
    candidate: Optional[CandidateMask],
    locked_tool: Optional[LockedToolMask],
    active_hand: Optional[dict],
    result: dict,
) -> None:
    writer.writerow(
        {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "phase": phase,
            "state": result["state"],
            "human_grasped_tool": result["human_grasped_tool"],
            "contact_state": result["contact_state"],
            "ml_raw_state": result["ml_raw_state"],
            "ml_stable_state": result["ml_stable_state"],
            "ml_confidence": _format_optional_float(result["ml_confidence"]),
            "ml_grasp_confirmed": result["ml_grasp_confirmed"],
            "ml_required": result["ml_required"],
            "ml_confidence_ok": result["ml_confidence_ok"],
            "contact_support": result["contact_support"],
            "mask_proximity_ok": result["mask_proximity_ok"],
            "mask_near_or_contact": result["mask_near_or_contact"],
            "decision_mode": result["decision_mode"],
            "grasp_counter": result["grasp_counter"],
            "grasp_score": result["grasp_score"],
            "mask_contact_count": result["mask_contact_count"],
            "mask_grasp_confirmed": result["mask_grasp_confirmed"],
            "tool_source": tool_source,
            "mask_source": mask_source,
            "tool_label": detection.label if detection is not None else "",
            "tool_confidence": f"{detection.confidence:.3f}" if detection is not None else "",
            "candidate_roi": candidate.roi if candidate is not None else "",
            "locked_roi": locked_tool.roi if locked_tool is not None else "",
            "active_hand_index": active_hand["hand_index"] if active_hand is not None else "",
            "active_handedness": active_hand["handedness"] if active_hand is not None else "",
        }
    )


def _print_state(phase: str, tool_source: str, mask_source: str, result: dict) -> None:
    print(
        "phase={phase}, state={state}, tool={tool_source}, mask={mask_source}, grasp_counter={counter}, "
        "mask_contact={mask_contact}, near={near}, contact_support={contact_support}, ml={ml_state}, "
        "ml_conf={ml_conf}, mode={mode}, score={score}, "
        "human_grasped_tool={human}".format(
            phase=phase,
            state=result["state"],
            tool_source=tool_source,
            mask_source=mask_source,
            counter=result["grasp_counter"],
            mask_contact=result["mask_contact_count"],
            near=result["mask_near_or_contact"],
            contact_support=result["contact_support"],
            ml_state=result["ml_stable_state"],
            ml_conf=_format_optional_float(result["ml_confidence"]),
            mode=result["decision_mode"],
            score=result["grasp_score"],
            human=result["human_grasped_tool"],
        )
    )


def _mask_to_rect(mask: np.ndarray) -> Optional[Rect]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return (x, y, x + width, y + height)


def _expand_and_clip_rect(rect: Rect, margin: int, shape: tuple[int, int]) -> Rect:
    x1, y1, x2, y2 = rect
    return _clip_rect((x1 - margin, y1 - margin, x2 + margin, y2 + margin), shape[1], shape[0])


def _expand_rect_unclipped(rect: Rect, margin: int) -> Rect:
    x1, y1, x2, y2 = rect
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _rect_center(rect: Rect) -> tuple[int, int]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _rect_mask(rect: Rect, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = _clip_rect(rect, shape[1], shape[0])
    mask[y1:y2, x1:x2] = True
    return mask


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def _clip_rect(rect: Rect, width: int, height: int) -> Rect:
    x1, y1, x2, y2 = rect
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "HUMAN_GRASPED_TOOL":
        return (0, 255, 0)
    if state in {"ML_GRASP_NO_CONTACT_SUPPORT", "ML_GRASP_WITH_CONTACT_SUPPORT"}:
        return (0, 180, 255)
    if state == "ML_GRASP_LOW_CONF":
        return (0, 220, 255)
    if state == "ML_GRASP_BUFFER_NOT_RAW":
        return (0, 200, 255)
    if state == "ML_GRASP_NOT_NEAR_TOOL":
        return (0, 140, 255)
    if state == "MASK_CONTACT_NO_ML_GRASP":
        return (255, 200, 0)
    if state == "CONTACT_WITHOUT_ML_GRASP":
        return (255, 160, 0)
    if state == "GRASP_CANDIDATE":
        return (0, 255, 255)
    if state == "HAND_NEAR_TOOL":
        return (0, 180, 255)
    if state == "HAND_DETECTED":
        return (255, 200, 0)
    return (220, 220, 220)


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
