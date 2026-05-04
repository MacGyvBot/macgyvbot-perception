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
from tool_detector import DEFAULT_MODEL_PATH, DEFAULT_TOOL_CLASSES, ToolDetection, ToolDetector
from utils import build_depth_grasp_info, draw_text, point_to_rect_distance, save_screenshot

CAMERA_INDEX = 0
WINDOW_NAME = "Hand Grasp Detection"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


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
    csv_file = None
    csv_writer: Optional[csv.DictWriter] = None
    previous_state: Optional[str] = None
    last_log_time = 0.0
    previous_frame_time = time.perf_counter()

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

            tool_detection = tool_detector.detect(frame) if tool_detector is not None else None
            if tool_detection is not None:
                tool_roi = tool_detection.roi
                tool_source = "YOLO"
            else:
                tool_roi = None
                tool_source = "NO_TOOL"

            hand_infos = hand_detector.detect_all(frame)
            active_hand = _select_active_hand(hand_infos, tool_roi)
            hand_for_state = active_hand if active_hand is not None else (hand_infos[0] if hand_infos else None)
            depth_info = _build_depth_info(args, hand_for_state, tool_roi, depth_mm)
            result = grasp_detector.update(hand_for_state, tool_roi, depth_info)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_frame_time, 1e-6)
            previous_frame_time = now

            _draw_overlay(frame, hand_infos, active_hand, tool_roi, tool_detection, tool_source, result, fps, args.depth_source)
            cv2.imshow(WINDOW_NAME, frame)

            _write_csv_row(csv_writer, result, tool_detection, tool_source, active_hand)
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
    parser.add_argument("--yolo-model", default=DEFAULT_MODEL_PATH, help="YOLO model path or built-in model name.")
    parser.add_argument(
        "--tool-classes",
        default=",".join(DEFAULT_TOOL_CLASSES),
        help="Comma-separated YOLO class names to use as tools. Use an empty string to accept any class.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.20, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--depth-source", choices=("none", "realsense"), default="none", help="Optional depth camera source.")
    parser.add_argument("--depth-diff-threshold-mm", type=float, default=50.0, help="Max hand-tool depth difference for depth contact.")
    parser.add_argument("--depth-min-contact-landmarks", type=int, default=2, help="Minimum depth-confirmed hand landmarks.")
    parser.add_argument("--realsense-width", type=int, default=640, help="RealSense color/depth stream width.")
    parser.add_argument("--realsense-height", type=int, default=480, help="RealSense color/depth stream height.")
    parser.add_argument("--realsense-fps", type=int, default=30, help="RealSense stream FPS.")
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


def _create_tool_detector(args: Namespace) -> Optional[ToolDetector]:
    target_classes = [name.strip() for name in args.tool_classes.split(",") if name.strip()]
    try:
        detector = ToolDetector(
            model_path=args.yolo_model,
            target_classes=target_classes,
            confidence_threshold=args.yolo_conf,
            image_size=args.yolo_imgsz,
        )
    except Exception as exc:
        print(f"WARNING: Failed to initialize YOLO detector: {exc}")
        print("WARNING: Tool ROI will be unavailable until YOLO initializes successfully.")
        return None

    if target_classes:
        print(f"INFO: YOLO enabled. model={args.yolo_model}, classes={target_classes}")
    else:
        print(f"INFO: YOLO enabled. model={args.yolo_model}, classes=ANY")
    return detector


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
        "depth_grasp_confirmed={depth_grasp_confirmed}, human_grasped_tool={human_grasped_tool}".format(
            state=result["state"],
            grasp_counter=result["grasp_counter"],
            pinch_distance=_format_optional_float(result["pinch_distance"]),
            palm_to_tool_distance=_format_optional_float(result["palm_to_tool_distance"]),
            contact_count=result["contact_count"],
            overlap=_format_optional_float(result["hand_tool_overlap_ratio"]),
            grasp_score=result["grasp_score"],
            depth_contact_count=result["depth_contact_count"],
            depth_grasp_confirmed=result["depth_grasp_confirmed"],
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
    result: dict,
    fps: float,
    depth_source: str,
) -> None:
    state = result["state"]
    roi_color = (0, 255, 0) if result["human_grasped_tool"] else (255, 120, 0)

    if tool_roi is not None and tool_detection is not None:
        x1, y1, x2, y2 = tool_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
        roi_label = f"{tool_detection.label} {tool_detection.confidence:.2f}"
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
    draw_text(frame, f"depth_contact: {result['depth_contact_count']}", (20, 272), scale=0.6)
    draw_text(frame, f"depth_diff: {_format_optional_float(result['min_hand_tool_depth_diff_mm'])}mm", (20, 302), scale=0.6)
    draw_text(frame, f"hands: {len(hand_infos)}", (20, 332), scale=0.6)
    draw_text(frame, f"tool_source: {tool_source}", (20, 362), scale=0.6)
    draw_text(frame, f"depth_source: {depth_source}", (20, 392), scale=0.6)
    draw_text(frame, f"FPS: {fps:.1f}", (20, 422), scale=0.6)


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
