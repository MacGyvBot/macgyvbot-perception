"""
Hand-grasp detection server (FastAPI + WebSocket).

Protocol:
  client --(JPEG bytes)-->          server  : frame for inference
  client --(JSON text {"cmd":...})--> server : control command
  server --(JSON text {...})-->     client  : per-frame inference result

Per-connection state (HandDetector, GraspDetector, ToolMaskTracker) is
created fresh for each WebSocket connection and destroyed on disconnect.
ToolDetector and SamSegmenter are loaded once at startup and shared.

Run:
    python src/main_server.py [--port 8765] [--sam-enabled] [--detector-source yolo]

Control commands from client:
    {"cmd": "reset"}                        -- reset grasp counter + mask tracker
    {"cmd": "set_roi", "roi": [x1,y1,x2,y2]} -- set manual tool ROI
    {"cmd": "clear_roi"}                    -- clear manual ROI
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from depth_camera import RealSenseDepthCamera
from grasp_detector import GraspDetector
from hand_detector import HandDetector
from sam_segmenter import SAM_TYPE_MOBILE, SAM_TYPE_SAM, SamSegmenter
from tool_detector import DEFAULT_MODEL_PATH, DEFAULT_TOOL_CLASSES, DEFAULT_YOLO_DEVICE, ToolDetection, ToolDetector
from tool_mask_tracker import MASK_OCCLUDED, ToolMaskTracker
from utils import build_depth_grasp_info, build_mask_grasp_info, point_to_rect_distance

DEFAULT_SAM_CHECKPOINT = str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "mobile_sam.pt")
DEFAULT_SAM_CHECKPOINTS = {
    "mobile_sam": str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "mobile_sam.pt"),
    "sam_vit_b":  str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "sam_vit_b.pth"),
    "sam_vit_l":  str(Path(__file__).resolve().parents[1] / "ckeckpoint" / "sam_vit_l.pth"),
}
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _encode_mask(mask: Optional[np.ndarray]) -> Optional[str]:
    """Encode boolean mask as base64 PNG for transport."""
    if mask is None:
        return None
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode()


def _serialize_hand_infos(hand_infos: list[dict]) -> list[dict]:
    """Convert numpy points to JSON-serializable lists."""
    out = []
    for h in hand_infos:
        out.append({
            "hand_index": h["hand_index"],
            "handedness": h["handedness"],
            "palm_center": list(h["palm_center"]),
            "thumb_tip": list(h["thumb_tip"]),
            "index_tip": list(h["index_tip"]),
            "landmarks": {str(k): list(v) for k, v in h["landmarks"].items()},
        })
    return out


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.1f}"


def _open_csv_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"grasp_log_{timestamp}.csv"
    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp", "state", "grasp_counter", "pinch_distance",
            "palm_to_tool_distance", "min_landmark_to_tool_distance",
            "contact_count", "hand_tool_overlap_ratio", "grasp_score",
            "depth_available", "tool_depth_mm", "min_hand_tool_depth_diff_mm",
            "depth_contact_count", "depth_grasp_confirmed",
            "mask_available", "mask_contact_count", "hand_mask_overlap_ratio",
            "mask_grasp_confirmed", "tool_mask_state", "tool_mask_missing_frames",
            "tool_mask_source", "human_grasped_tool",
            "active_hand_index", "active_handedness",
            "tool_source", "tool_label", "tool_confidence",
        ],
    )
    writer.writeheader()
    print(f"[server] CSV log: {path}")
    return csv_file, writer


def _write_csv_row(writer, result, tool_detection, tool_source, mask_state, active_hand) -> None:
    writer.writerow({
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
    })


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="grasp-detection-server")

    # Auto-fill SAM checkpoint/model-type when user picks --sam-type without explicit --sam-checkpoint
    if args.sam_type == SAM_TYPE_SAM and args.sam_checkpoint == DEFAULT_SAM_CHECKPOINT:
        # User switched to 'sam' but didn't override checkpoint — pick vit_b by default
        if args.sam_model_type == "vit_t":
            args.sam_model_type = "vit_b"
        args.sam_checkpoint = DEFAULT_SAM_CHECKPOINTS.get(f"sam_{args.sam_model_type}", args.sam_checkpoint)
        print(f"[server] auto-selected checkpoint: {args.sam_checkpoint} (model_type={args.sam_model_type})")

    # Shared stateless components (loaded once)
    tool_detector: Optional[ToolDetector] = None
    if args.detector_source == "yolo":
        target_classes = [c.strip() for c in args.tool_classes.split(",") if c.strip()]
        if not target_classes:
            print("ERROR: --detector-source yolo requires --tool-classes.")
            sys.exit(1)
        try:
            tool_detector = ToolDetector(
                model_path=args.yolo_model,
                target_classes=target_classes,
                confidence_threshold=args.yolo_conf,
                iou_threshold=args.yolo_iou,
                image_size=args.yolo_imgsz,
                device=args.yolo_device,
            )
            print(f"[server] YOLO ready. classes={target_classes} device={args.yolo_device}")
        except Exception as exc:
            print(f"[server] WARNING: YOLO init failed: {exc}")
    else:
        print(f"[server] detector_source={args.detector_source}, YOLO disabled.")

    sam_segmenter: Optional[SamSegmenter] = None
    if args.sam_enabled:
        try:
            sam_segmenter = SamSegmenter(
                checkpoint_path=args.sam_checkpoint,
                model_type=args.sam_model_type,
                device=args.sam_device,
                sam_type=args.sam_type,
            )
            print(f"[server] SAM ready. sam_type={args.sam_type} model_type={args.sam_model_type} device={args.sam_device}")
        except Exception as exc:
            print(f"[server] WARNING: SAM init failed: {exc}")

    depth_camera: Optional[RealSenseDepthCamera] = None
    if args.depth_source == "realsense":
        try:
            depth_camera = RealSenseDepthCamera(
                width=args.realsense_width,
                height=args.realsense_height,
                fps=args.realsense_fps,
            )
            print("[server] RealSense depth source enabled.")
        except Exception as exc:
            print(f"[server] ERROR: Failed to initialize RealSense: {exc}")
            sys.exit(1)

    print("[server] ready — waiting for connections.")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "detector_source": args.detector_source,
            "sam_enabled": sam_segmenter is not None,
        }

    @app.websocket("/infer")
    async def infer_ws(ws: WebSocket) -> None:
        await ws.accept()
        peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        print(f"[server] connected: {peer}")

        # Per-connection stateful objects
        hand_detector = HandDetector(max_num_hands=args.max_hands)
        grasp_detector = GraspDetector()
        tool_mask_tracker = ToolMaskTracker(
            max_missing_frames=args.mask_max_missing_frames,
            min_lock_area=args.mask_min_lock_area,
            smoothing_alpha=args.mask_smoothing_alpha,
        )
        manual_roi: Optional[tuple] = None
        manual_roi_pending = False
        n = 0
        csv_file, csv_writer = _open_csv_logger()
        loop = asyncio.get_event_loop()

        def _sync_infer(frame: np.ndarray, depth_mm, _manual_roi, _manual_roi_pending: bool):
            """Run all heavy inference synchronously (called via run_in_executor)."""
            # Tool detection
            tool_detection: Optional[ToolDetection] = None
            if args.detector_source == "yolo" and tool_detector is not None:
                tool_detection = tool_detector.detect(frame)
            elif args.detector_source == "manual" and _manual_roi is not None and _manual_roi_pending:
                tool_detection = ToolDetection(roi=_manual_roi, label="manual_roi", confidence=1.0)

            # SAM — set_image once per frame, predict_box per bbox (yolo_mobilesam_server.py approach)
            detected_mask = None
            if sam_segmenter is not None and tool_detection is not None:
                try:
                    sam_segmenter.set_image(frame)
                    detected_mask = sam_segmenter.predict_box(tool_detection.roi)
                except Exception as exc:
                    print(f"[server] SAM warning: {exc}")

            # Mask tracker
            mask_state = tool_mask_tracker.update(
                detected_roi=tool_detection.roi if tool_detection else None,
                detected_mask=detected_mask,
                frame_shape=frame.shape[:2],
            )
            tool_roi = mask_state.roi
            tool_source = mask_state.source

            # Hand detection
            hand_infos = hand_detector.detect_all(frame)

            # Active hand (closest to tool)
            active_hand = None
            if hand_infos and tool_roi is not None:
                active_hand = min(
                    hand_infos,
                    key=lambda h: point_to_rect_distance(h["palm_center"], tool_roi),
                )
            hand_for_state = active_hand if active_hand is not None else (hand_infos[0] if hand_infos else None)

            # Depth grasp info
            depth_info = None
            if depth_mm is not None and hand_for_state is not None and tool_roi is not None:
                depth_info = build_depth_grasp_info(
                    hand_info=hand_for_state,
                    tool_roi=tool_roi,
                    depth_mm=depth_mm,
                    depth_diff_threshold_mm=args.depth_diff_threshold_mm,
                    min_depth_contact_landmarks=args.depth_min_contact_landmarks,
                )

            # Mask grasp info
            mask_info = None
            if hand_for_state is not None and mask_state.mask is not None:
                mask_info = build_mask_grasp_info(
                    hand_info=hand_for_state,
                    tool_mask=mask_state.mask,
                    contact_radius=args.mask_contact_radius,
                    min_mask_contact_landmarks=args.mask_min_contact_landmarks,
                )

            # Grasp state
            result = grasp_detector.update(hand_for_state, tool_roi, depth_info, mask_info)

            return result, hand_infos, active_hand, mask_state, tool_detection, tool_source

        try:
            while True:
                message = await ws.receive()

                if message["type"] == "websocket.disconnect":
                    break

                # --- binary: JPEG frame ---
                if message.get("bytes"):
                    data = message["bytes"]
                    t0 = time.time()

                    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        await ws.send_text(json.dumps({"error": "decode_failed"}))
                        continue

                    # Depth (RealSense on server side)
                    depth_mm = None
                    if depth_camera is not None:
                        depth_frame = depth_camera.read()
                        if depth_frame is not None:
                            depth_mm = depth_frame.depth_mm

                    # Run inference in thread pool (releases event loop during GPU work)
                    (result, hand_infos, active_hand, mask_state, tool_detection, tool_source) = \
                        await loop.run_in_executor(
                            None, _sync_infer, frame, depth_mm, manual_roi, manual_roi_pending
                        )

                    if args.detector_source == "manual" and manual_roi_pending:
                        manual_roi_pending = False

                    _write_csv_row(csv_writer, result, tool_detection, tool_source, mask_state, active_hand)

                    is_occluded = mask_state.state == MASK_OCCLUDED
                    payload = {
                        "result": result,
                        "hand_infos": _serialize_hand_infos(hand_infos),
                        "active_hand_index": active_hand["hand_index"] if active_hand else None,
                        "tool_roi": None if is_occluded else (list(mask_state.roi) if mask_state.roi else None),
                        "tool_detection": None if is_occluded else ({
                            "label": tool_detection.label,
                            "confidence": tool_detection.confidence,
                            "roi": list(tool_detection.roi),
                        } if tool_detection else None),
                        "tool_source": tool_source,
                        "mask_state": {
                            "state": mask_state.state,
                            "source": mask_state.source,
                            "missing_frames": mask_state.missing_frames,
                        },
                        "mask_b64": _encode_mask(mask_state.mask),
                        "infer_ms": (time.time() - t0) * 1000.0,
                    }
                    await ws.send_text(json.dumps(payload))
                    n += 1
                    if n % 100 == 0:
                        print(f"[server] {peer} frames={n} infer_ms={payload['infer_ms']:.1f}")

                # --- text: control command ---
                elif message.get("text"):
                    try:
                        cmd = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    action = cmd.get("cmd")
                    if action == "reset":
                        grasp_detector.reset()
                        tool_mask_tracker.reset()
                        print(f"[server] {peer} reset")
                    elif action == "set_roi":
                        roi = cmd.get("roi")
                        if roi and len(roi) == 4:
                            manual_roi = tuple(int(v) for v in roi)
                            manual_roi_pending = True
                            tool_mask_tracker.reset()
                            grasp_detector.reset()
                            print(f"[server] {peer} manual ROI: {manual_roi}")
                    elif action == "clear_roi":
                        manual_roi = None
                        manual_roi_pending = False
                        tool_mask_tracker.reset()
                        grasp_detector.reset()
                        print(f"[server] {peer} manual ROI cleared")

        except WebSocketDisconnect:
            print(f"[server] disconnected: {peer} (frames={n})")
        except Exception as exc:
            print(f"[server] error on {peer}: {exc!r}")
            try:
                await ws.close()
            except Exception:
                pass
        finally:
            hand_detector.close()
            csv_file.close()

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grasp detection inference server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--detector-source", choices=("yolo", "manual", "none"), default="yolo")
    p.add_argument("--yolo-model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--yolo-device", default=DEFAULT_YOLO_DEVICE)
    p.add_argument("--tool-classes", default=",".join(DEFAULT_TOOL_CLASSES))
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.45)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--no-sam", dest="sam_enabled", action="store_false", help="Disable SAM segmentation.")
    p.set_defaults(sam_enabled=True)
    p.add_argument(
        "--sam-type",
        choices=(SAM_TYPE_MOBILE, SAM_TYPE_SAM),
        default=SAM_TYPE_MOBILE,
        help="SAM backend: 'mobile_sam' (vit_t, fast) or 'sam' (vit_b/vit_l/vit_h, accurate).",
    )
    p.add_argument(
        "--sam-model-type",
        default="vit_t",
        help="Model type: 'vit_t' for mobile_sam; 'vit_b'/'vit_l'/'vit_h' for sam.",
    )
    p.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_SAM_CHECKPOINT,
        help=f"Checkpoint path. Defaults per --sam-type: {DEFAULT_SAM_CHECKPOINTS}",
    )
    p.add_argument("--sam-device", default="cuda")
    p.add_argument("--mask-contact-radius", type=int, default=6)
    p.add_argument("--mask-min-contact-landmarks", type=int, default=2)
    p.add_argument("--mask-max-missing-frames", type=int, default=-1)
    p.add_argument("--mask-min-lock-area", type=int, default=100)
    p.add_argument("--mask-smoothing-alpha", type=float, default=0.7)
    p.add_argument("--depth-source", choices=("none", "realsense"), default="none")
    p.add_argument("--depth-diff-threshold-mm", type=float, default=50.0)
    p.add_argument("--depth-min-contact-landmarks", type=int, default=2)
    p.add_argument("--realsense-width", type=int, default=640)
    p.add_argument("--realsense-height", type=int, default=480)
    p.add_argument("--realsense-fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
