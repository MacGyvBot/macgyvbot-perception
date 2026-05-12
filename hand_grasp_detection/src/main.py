from __future__ import annotations

import csv
import sys
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from depth_camera import RealSenseDepthCamera
from grasp_detector import GraspDetector
from hand_detector import HandDetector
from sam_segmenter import SAM_TYPE_MOBILE, SAM_TYPE_SAM, SamSegmenter
from tool_detector import DEFAULT_MODEL_PATH, DEFAULT_TOOL_CLASSES, DEFAULT_YOLO_DEVICE, ToolDetection, ToolDetector
from tool_mask_tracker import MASK_OCCLUDED, ToolMaskTracker
from utils import build_depth_grasp_info, build_mask_grasp_info, draw_text, point_to_rect_distance, save_screenshot

CAMERA_INDEX = 0
WINDOW_NAME = "Hand Grasp Detection"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_SAM_CHECKPOINT = str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "mobile_sam.pt")
DEFAULT_SAM_CHECKPOINTS = {
    "mobile_sam": str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "mobile_sam.pt"),
    "sam_vit_b":  str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "sam_vit_b.pth"),
    "sam_vit_l":  str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "sam_vit_l.pth"),
}


def main() -> int:
    args = _parse_args()
    cap = None
    depth_camera = None
    if args.depth_source == "realsense":
        try:
            depth_camera = RealSenseDepthCamera(width=args.realsense_width, height=args.realsense_height, fps=args.realsense_fps)
            print("INFO: RealSense depth source enabled.")
        except Exception as exc:
            print(f"ERROR: Failed to initialize RealSense depth camera: {exc}")
            return 1
    else:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"ERROR: Failed to open camera index {args.camera_index}.")
            return 1

    hand_detector = HandDetector(max_num_hands=args.max_hands)
    grasp_detector = GraspDetector()
    tool_detector = _create_tool_detector(args)
    sam_segmenter = _create_sam_segmenter(args)
    tool_mask_tracker = ToolMaskTracker(
        max_missing_frames=args.mask_max_missing_frames,
        min_lock_area=args.mask_min_lock_area,
        smoothing_alpha=args.mask_smoothing_alpha,
    )
    csv_file = None
    csv_writer: Optional[csv.DictWriter] = None
    previous_state: Optional[str] = None
    last_log_time = 0.0
    previous_frame_time = time.perf_counter()
    manual_roi: Optional[tuple[int, int, int, int]] = None
    manual_roi_pending = False

    try:
        csv_file, csv_writer = _open_csv_logger()

        while True:
            frame, depth_mm = _read_frame(cap, depth_camera)
            if frame is None:
                print("ERROR: Failed to read frame from camera source.")
                break

            frame = cv2.flip(frame, 1)
            if depth_mm is not None:
                depth_mm = cv2.flip(depth_mm, 1)

            selection_frame = frame.copy()
            tool_detection = _detect_tool(args, tool_detector, frame, manual_roi, manual_roi_pending)
            detected_mask = _segment_tool_mask(sam_segmenter, frame, tool_detection)
            if args.detector_source == "manual" and manual_roi_pending:
                manual_roi_pending = False
            mask_state = tool_mask_tracker.update(
                detected_roi=tool_detection.roi if tool_detection else None,
                detected_mask=detected_mask,
                frame_shape=frame.shape[:2],
            )
            tool_roi = mask_state.roi
            tool_source = mask_state.source

            hand_infos = hand_detector.detect_all(frame)
            active_hand = _select_active_hand(hand_infos, tool_roi)
            hand_for_state = active_hand if active_hand is not None else (hand_infos[0] if hand_infos else None)
            depth_info = _build_depth_info(args, hand_for_state, tool_roi, depth_mm)
            mask_info = _build_mask_info(args, hand_for_state, mask_state.mask)
            result = grasp_detector.update(hand_for_state, tool_roi, depth_info, mask_info)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_frame_time, 1e-6)
            previous_frame_time = now

            _draw_overlay(frame, hand_infos, active_hand, tool_roi, tool_detection, tool_source, mask_state, result, fps, args.depth_source)
            cv2.imshow(WINDOW_NAME, frame)

            _write_csv_row(csv_writer, result, tool_detection, tool_source, mask_state, active_hand)
            if result["state"] != previous_state or now - last_log_time >= 1.0:
                _print_state(result)
                previous_state = result["state"]
                last_log_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                grasp_detector.reset()
                print("INFO: Grasp counter reset.")
            if key == ord("s"):
                path = save_screenshot(frame, str(LOG_DIR))
                print(f"INFO: Screenshot saved to {path}")
            if key == ord("m"):
                selected_roi = _select_manual_roi(selection_frame)
                if selected_roi is not None:
                    manual_roi = selected_roi
                    manual_roi_pending = True
                    tool_mask_tracker.reset()
                    grasp_detector.reset()
                    print(f"INFO: Manual ROI set to {manual_roi}. Locked mask will be initialized on the next frame.")
            if key == ord("c"):
                manual_roi = None
                manual_roi_pending = False
                tool_mask_tracker.reset()
                grasp_detector.reset()
                print("INFO: Manual ROI and locked mask cleared.")

    finally:
        hand_detector.close()
        if cap is not None:
            cap.release()
        if depth_camera is not None:
            depth_camera.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    return 0


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="MacBook camera hand-tool grasp detection.")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="OpenCV camera index.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands MediaPipe should track.")
    parser.add_argument(
        "--detector-source",
        choices=("manual", "yolo", "none"),
        default="yolo",
        help="Tool initialization source.",
    )
    parser.add_argument("--yolo-model", default=DEFAULT_MODEL_PATH, help="YOLO model path or built-in model name.")
    parser.add_argument("--yolo-device", default=DEFAULT_YOLO_DEVICE, help="YOLO device, for example cuda, cpu, or mps.")
    parser.add_argument(
        "--tool-classes",
        default=",".join(DEFAULT_TOOL_CLASSES),
        help="Comma-separated YOLO class names to use as tools. YOLO mode requires at least one class.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IOU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--depth-source", choices=("none", "realsense"), default="none", help="Optional depth camera source.")
    parser.add_argument("--depth-diff-threshold-mm", type=float, default=50.0, help="Max hand-tool depth difference for depth contact.")
    parser.add_argument("--depth-min-contact-landmarks", type=int, default=2, help="Minimum depth-confirmed hand landmarks.")
    parser.add_argument("--realsense-width", type=int, default=640, help="RealSense color/depth stream width.")
    parser.add_argument("--realsense-height", type=int, default=480, help="RealSense color/depth stream height.")
    parser.add_argument("--realsense-fps", type=int, default=30, help="RealSense stream FPS.")
    parser.add_argument("--no-sam", dest="sam_enabled", action="store_false", help="Disable SAM segmentation.")
    parser.set_defaults(sam_enabled=True)
    parser.add_argument(
        "--sam-type",
        choices=(SAM_TYPE_MOBILE, SAM_TYPE_SAM),
        default=SAM_TYPE_MOBILE,
        help="SAM backend: 'mobile_sam' (vit_t, fast) or 'sam' (vit_b/vit_l/vit_h, accurate).",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_t",
        help="Model type: 'vit_t' for mobile_sam; 'vit_b'/'vit_l'/'vit_h' for sam.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_SAM_CHECKPOINT,
        help=f"Checkpoint path. Defaults per --sam-type: {DEFAULT_SAM_CHECKPOINTS}",
    )
    parser.add_argument("--sam-device", default="cuda", help="SAM device, for example cpu, cuda, or mps.")
    parser.add_argument("--mask-contact-radius", type=int, default=6, help="Pixel radius for landmark-to-mask contact.")
    parser.add_argument("--mask-min-contact-landmarks", type=int, default=2, help="Minimum landmarks touching locked mask.")
    parser.add_argument(
        "--mask-max-missing-frames",
        type=int,
        default=-1,
        help="Frames to keep locked mask after detector loss. Use -1 to keep it until cleared.",
    )
    parser.add_argument("--mask-min-lock-area", type=int, default=100, help="Minimum mask area required for locking.")
    parser.add_argument("--mask-smoothing-alpha", type=float, default=0.7, help="ROI smoothing alpha for locked mask tracking.")
    return parser.parse_args()


def _read_frame(cap, depth_camera: Optional[RealSenseDepthCamera]):
    if depth_camera is not None:
        depth_frame = depth_camera.read()
        if depth_frame is None:
            return None, None
        return depth_frame.color_bgr, depth_frame.depth_mm

    ok, frame = cap.read()
    if not ok:
        return None, None
    return frame, None


def _build_depth_info(
    args: Namespace,
    hand_info: Optional[dict],
    tool_roi: Optional[tuple[int, int, int, int]],
    depth_mm,
) -> Optional[dict]:
    if args.depth_source == "none" or depth_mm is None or hand_info is None or tool_roi is None:
        return None

    return build_depth_grasp_info(
        hand_info=hand_info,
        tool_roi=tool_roi,
        depth_mm=depth_mm,
        depth_diff_threshold_mm=args.depth_diff_threshold_mm,
        min_depth_contact_landmarks=args.depth_min_contact_landmarks,
    )


def _build_mask_info(args: Namespace, hand_info: Optional[dict], tool_mask) -> Optional[dict]:
    if hand_info is None or tool_mask is None:
        return None

    return build_mask_grasp_info(
        hand_info=hand_info,
        tool_mask=tool_mask,
        contact_radius=args.mask_contact_radius,
        min_mask_contact_landmarks=args.mask_min_contact_landmarks,
    )


def _detect_tool(
    args: Namespace,
    tool_detector: Optional[ToolDetector],
    frame,
    manual_roi: Optional[tuple[int, int, int, int]],
    manual_roi_pending: bool,
) -> Optional[ToolDetection]:
    if args.detector_source == "yolo":
        return tool_detector.detect(frame) if tool_detector is not None else None

    if args.detector_source == "manual" and manual_roi is not None and manual_roi_pending:
        return ToolDetection(roi=manual_roi, label="manual_roi", confidence=1.0)

    return None


def _create_tool_detector(args: Namespace) -> Optional[ToolDetector]:
    if args.detector_source != "yolo":
        print(f"INFO: YOLO disabled. detector_source={args.detector_source}")
        return None

    target_classes = [name.strip() for name in args.tool_classes.split(",") if name.strip()]
    if not target_classes:
        print("ERROR: --detector-source yolo requires --tool-classes with at least one class name.")
        print("ERROR: Example: python src/main.py --detector-source yolo --tool-classes scissors")
        return None

    try:
        detector = ToolDetector(
            model_path=args.yolo_model,
            target_classes=target_classes,
            confidence_threshold=args.yolo_conf,
            iou_threshold=args.yolo_iou,
            image_size=args.yolo_imgsz,
            device=args.yolo_device,
        )
    except Exception as exc:
        print(f"WARNING: Failed to initialize YOLO detector: {exc}")
        print("WARNING: Tool ROI will be unavailable until YOLO initializes successfully.")
        return None

    print(f"INFO: YOLO enabled. model={args.yolo_model}, classes={target_classes}, device={args.yolo_device}")
    return detector


def _select_manual_roi(frame) -> Optional[tuple[int, int, int, int]]:
    roi = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=True, fromCenter=False)
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        print("INFO: Manual ROI selection cancelled.")
        return None
    return (x, y, x + width, y + height)


def _create_sam_segmenter(args: Namespace) -> Optional[SamSegmenter]:
    if not args.sam_enabled:
        return None

    # Auto-fill checkpoint/model-type when user picks --sam-type without explicit --sam-checkpoint
    checkpoint = args.sam_checkpoint
    model_type = args.sam_model_type
    if args.sam_type == SAM_TYPE_SAM and checkpoint == DEFAULT_SAM_CHECKPOINT:
        if model_type == "vit_t":
            model_type = "vit_b"
        checkpoint = DEFAULT_SAM_CHECKPOINTS.get(f"sam_{model_type}", checkpoint)
        print(f"INFO: auto-selected SAM checkpoint: {checkpoint} (model_type={model_type})")

    try:
        segmenter = SamSegmenter(
            checkpoint_path=checkpoint,
            model_type=model_type,
            device=args.sam_device,
            sam_type=args.sam_type,
        )
    except Exception as exc:
        print(f"WARNING: Failed to initialize SAM segmenter: {exc}")
        return None

    print(f"INFO: SAM enabled. sam_type={args.sam_type}, model_type={model_type}, device={args.sam_device}")
    return segmenter


def _segment_tool_mask(
    sam_segmenter: Optional[SamSegmenter],
    frame,
    tool_detection: Optional[ToolDetection],
):
    if sam_segmenter is None or tool_detection is None:
        return None

    try:
        return sam_segmenter.segment(frame, tool_detection.roi)
    except Exception as exc:
        print(f"WARNING: SAM segmentation failed: {exc}")
        return None


def _select_active_hand(hand_infos: list[dict], tool_roi: Optional[tuple[int, int, int, int]]) -> Optional[dict]:
    if not hand_infos or tool_roi is None:
        return None

    return min(hand_infos, key=lambda hand_info: point_to_rect_distance(hand_info["palm_center"], tool_roi))


def _open_csv_logger() -> tuple[object, csv.DictWriter]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"grasp_log_{timestamp}.csv"
    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp",
            "state",
            "grasp_counter",
            "pinch_distance",
            "palm_to_tool_distance",
            "min_landmark_to_tool_distance",
            "contact_count",
            "hand_tool_overlap_ratio",
            "grasp_score",
            "depth_available",
            "tool_depth_mm",
            "min_hand_tool_depth_diff_mm",
            "depth_contact_count",
            "depth_grasp_confirmed",
            "mask_available",
            "mask_contact_count",
            "hand_mask_overlap_ratio",
            "mask_grasp_confirmed",
            "tool_mask_state",
            "tool_mask_missing_frames",
            "tool_mask_source",
            "human_grasped_tool",
            "active_hand_index",
            "active_handedness",
            "tool_source",
            "tool_label",
            "tool_confidence",
        ],
    )
    writer.writeheader()
    print(f"INFO: CSV log file: {path}")
    return csv_file, writer


def _write_csv_row(
    csv_writer: Optional[csv.DictWriter],
    result: dict,
    tool_detection: Optional[ToolDetection],
    tool_source: str,
    mask_state,
    active_hand: Optional[dict] = None,
) -> None:
    if csv_writer is None:
        return

    csv_writer.writerow(
        {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "state": result["state"],
            "grasp_counter": result["grasp_counter"],
            "pinch_distance": _format_optional_float(result["pinch_distance"]),
            "palm_to_tool_distance": _format_optional_float(result["palm_to_tool_distance"]),
            "min_landmark_to_tool_distance": _format_optional_float(result["min_landmark_to_tool_distance"]),
            "contact_count": result["contact_count"],
            "hand_tool_overlap_ratio": _format_optional_float(result["hand_tool_overlap_ratio"]),
            "grasp_score": result["grasp_score"],
            "depth_available": result["depth_available"],
            "tool_depth_mm": _format_optional_float(result["tool_depth_mm"]),
            "min_hand_tool_depth_diff_mm": _format_optional_float(result["min_hand_tool_depth_diff_mm"]),
            "depth_contact_count": result["depth_contact_count"],
            "depth_grasp_confirmed": result["depth_grasp_confirmed"],
            "mask_available": result["mask_available"],
            "mask_contact_count": result["mask_contact_count"],
            "hand_mask_overlap_ratio": _format_optional_float(result["hand_mask_overlap_ratio"]),
            "mask_grasp_confirmed": result["mask_grasp_confirmed"],
            "tool_mask_state": mask_state.state,
            "tool_mask_missing_frames": mask_state.missing_frames,
            "tool_mask_source": mask_state.source,
            "human_grasped_tool": result["human_grasped_tool"],
            "active_hand_index": active_hand["hand_index"] if active_hand else "",
            "active_handedness": active_hand["handedness"] if active_hand else "",
            "tool_source": tool_source,
            "tool_label": tool_detection.label if tool_detection else "",
            "tool_confidence": _format_optional_float(tool_detection.confidence if tool_detection else None),
        }
    )


def _print_state(result: dict) -> None:
    print(
        "state={state}, grasp_counter={grasp_counter}, pinch_distance={pinch_distance}, "
        "palm_to_tool_distance={palm_to_tool_distance}, contact_count={contact_count}, "
        "overlap={overlap}, grasp_score={grasp_score}, depth_contact_count={depth_contact_count}, "
        "depth_grasp_confirmed={depth_grasp_confirmed}, mask_contact_count={mask_contact_count}, "
        "mask_grasp_confirmed={mask_grasp_confirmed}, human_grasped_tool={human_grasped_tool}".format(
            state=result["state"],
            grasp_counter=result["grasp_counter"],
            pinch_distance=_format_optional_float(result["pinch_distance"]),
            palm_to_tool_distance=_format_optional_float(result["palm_to_tool_distance"]),
            contact_count=result["contact_count"],
            overlap=_format_optional_float(result["hand_tool_overlap_ratio"]),
            grasp_score=result["grasp_score"],
            depth_contact_count=result["depth_contact_count"],
            depth_grasp_confirmed=result["depth_grasp_confirmed"],
            mask_contact_count=result["mask_contact_count"],
            mask_grasp_confirmed=result["mask_grasp_confirmed"],
            human_grasped_tool=result["human_grasped_tool"],
        )
    )


def _draw_overlay(
    frame,
    hand_infos: list[dict],
    active_hand: Optional[dict],
    tool_roi: Optional[tuple[int, int, int, int]],
    tool_detection: Optional[ToolDetection],
    tool_source: str,
    mask_state,
    result: dict,
    fps: float,
    depth_source: str,
) -> None:
    state = result["state"]
    roi_color = (0, 255, 0) if result["human_grasped_tool"] else (255, 120, 0)

    if mask_state.mask is not None:
        overlay = frame.copy()
        overlay[mask_state.mask] = (0, 160, 255)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    if tool_roi is not None and mask_state.state != MASK_OCCLUDED:
        x1, y1, x2, y2 = tool_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
        if tool_detection is not None:
            roi_label = f"{tool_detection.label} {tool_detection.confidence:.2f}"
        else:
            roi_label = mask_state.state
        draw_text(frame, roi_label, (x1, max(24, y1 - 10)), roi_color, scale=0.55)

    active_hand_index = active_hand["hand_index"] if active_hand else None

    for hand_info in hand_infos:
        is_active = hand_info["hand_index"] == active_hand_index
        point_color = (0, 255, 255) if is_active else (255, 200, 0)
        point_radius = 4 if is_active else 3
        landmarks = hand_info["landmarks"]
        for point in landmarks.values():
            cv2.circle(frame, point, point_radius, point_color, -1)

        palm_center = hand_info["palm_center"]
        thumb_tip = hand_info["thumb_tip"]
        index_tip = hand_info["index_tip"]
        cv2.circle(frame, palm_center, 7 if is_active else 5, (0, 255, 255) if is_active else (180, 180, 180), -1)
        cv2.circle(frame, thumb_tip, 6 if is_active else 4, (255, 0, 255), -1)
        cv2.circle(frame, index_tip, 6 if is_active else 4, (0, 255, 255), -1)
        cv2.line(frame, thumb_tip, index_tip, (255, 255, 0) if is_active else (160, 160, 160), 2 if is_active else 1)
        label = f"{hand_info['hand_index']} {hand_info['handedness']}"
        if is_active:
            label += " ACTIVE"
        draw_text(frame, label, (palm_center[0] + 8, palm_center[1] - 8), point_color, scale=0.45, thickness=1)

    draw_text(frame, f"State: {state}", (20, 32), _state_color(state))
    draw_text(frame, f"human_grasped_tool: {result['human_grasped_tool']}", (20, 62), _state_color(state), scale=0.6)
    draw_text(frame, f"grasp_counter: {result['grasp_counter']}", (20, 92), scale=0.6)
    draw_text(frame, f"pinch_distance: {_format_optional_float(result['pinch_distance'])}", (20, 122), scale=0.6)
    draw_text(frame, f"palm_to_tool: {_format_optional_float(result['palm_to_tool_distance'])}", (20, 152), scale=0.6)
    draw_text(frame, f"contact_count: {result['contact_count']}", (20, 182), scale=0.6)
    draw_text(frame, f"overlap: {_format_optional_float(result['hand_tool_overlap_ratio'])}", (20, 212), scale=0.6)
    draw_text(frame, f"grasp_score: {result['grasp_score']}", (20, 242), scale=0.6)
    draw_text(frame, f"mask_contact: {result['mask_contact_count']}", (20, 272), scale=0.6)
    draw_text(frame, f"mask_state: {mask_state.state}", (20, 302), scale=0.6)
    draw_text(frame, f"depth_contact: {result['depth_contact_count']}", (20, 332), scale=0.6)
    draw_text(frame, f"depth_diff: {_format_optional_float(result['min_hand_tool_depth_diff_mm'])}mm", (20, 362), scale=0.6)
    draw_text(frame, f"hands: {len(hand_infos)}", (20, 392), scale=0.6)
    draw_text(frame, f"tool_source: {tool_source}", (20, 422), scale=0.6)
    draw_text(frame, f"depth_source: {depth_source}", (20, 452), scale=0.6)
    draw_text(frame, f"FPS: {fps:.1f}", (20, 482), scale=0.6)


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "HUMAN_GRASPED_TOOL":
        return (0, 255, 0)
    if state == "GRASP_CANDIDATE":
        return (0, 255, 255)
    if state == "HAND_NEAR_TOOL":
        return (0, 180, 255)
    if state == "HAND_DETECTED":
        return (255, 200, 0)
    if state == "TOOL_NOT_DETECTED":
        return (180, 180, 180)
    return (200, 200, 200)


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.1f}"


if __name__ == "__main__":
    sys.exit(main())
