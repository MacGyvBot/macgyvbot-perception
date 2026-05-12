"""
Local camera client for remote grasp detection.

Pipeline:
    camera --> JPEG encode --(WebSocket binary)--> server
    server --(WebSocket JSON)-->  decode + draw --> cv2.imshow

The render loop applies the same overlay as main.py.
Control commands are sent to the server via the same WebSocket connection.

Dependencies (client-only): opencv-python, numpy, websockets

Run:
    python src/main_client.py --url ws://<server-ip>:8765/infer

Keyboard shortcuts:
    q  quit
    r  reset grasp counter (sent to server)
    s  save screenshot locally
    m  select manual tool ROI (sent to server)
    c  clear manual ROI (sent to server)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import websockets

WINDOW_NAME = "Hand Grasp Detection (Remote)"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


# ---------------------------------------------------------------------------
# Drawing helpers (self-contained, no dependency on server-side modules)
# ---------------------------------------------------------------------------

def _draw_text(
    frame,
    text: str,
    pos: tuple,
    color: tuple = (255, 255, 255),
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _state_color(state: str) -> tuple:
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


def _fmt(value) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _decode_mask(mask_b64: Optional[str]) -> Optional[np.ndarray]:
    if mask_b64 is None:
        return None
    try:
        buf = base64.b64decode(mask_b64)
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        return img > 0
    except Exception:
        return None


def _draw_overlay(frame, state: dict, fps: float) -> None:
    result = state.get("result") or {}
    hand_infos = state.get("hand_infos") or []
    active_hand_index = state.get("active_hand_index")
    tool_roi = state.get("tool_roi")
    tool_detection = state.get("tool_detection")
    tool_source = state.get("tool_source", "none")
    mask_state = state.get("mask_state") or {}
    mask = state.get("mask")

    state_str = result.get("state", "NO_DATA")
    roi_color = (0, 255, 0) if result.get("human_grasped_tool") else (255, 120, 0)

    if mask is not None:
        overlay = frame.copy()
        overlay[mask] = (0, 160, 255)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    if tool_roi is not None:
        x1, y1, x2, y2 = (int(v) for v in tool_roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
        if tool_detection is not None:
            roi_label = f"{tool_detection['label']} {tool_detection['confidence']:.2f}"
        else:
            roi_label = mask_state.get("state", "")
        _draw_text(frame, roi_label, (x1, max(24, y1 - 10)), roi_color, scale=0.55)

    for hand_info in hand_infos:
        is_active = hand_info["hand_index"] == active_hand_index
        point_color = (0, 255, 255) if is_active else (255, 200, 0)
        point_radius = 4 if is_active else 3

        for pt in hand_info["landmarks"].values():
            cv2.circle(frame, (int(pt[0]), int(pt[1])), point_radius, point_color, -1)

        palm = (int(hand_info["palm_center"][0]), int(hand_info["palm_center"][1]))
        thumb = (int(hand_info["thumb_tip"][0]), int(hand_info["thumb_tip"][1]))
        index = (int(hand_info["index_tip"][0]), int(hand_info["index_tip"][1]))
        cv2.circle(frame, palm, 7 if is_active else 5, (0, 255, 255) if is_active else (180, 180, 180), -1)
        cv2.circle(frame, thumb, 6 if is_active else 4, (255, 0, 255), -1)
        cv2.circle(frame, index, 6 if is_active else 4, (0, 255, 255), -1)
        cv2.line(frame, thumb, index, (255, 255, 0) if is_active else (160, 160, 160), 2 if is_active else 1)
        label = f"{hand_info['hand_index']} {hand_info['handedness']}"
        if is_active:
            label += " ACTIVE"
        _draw_text(frame, label, (palm[0] + 8, palm[1] - 8), point_color, scale=0.45, thickness=1)

    _draw_text(frame, f"State: {state_str}", (20, 32), _state_color(state_str))
    _draw_text(frame, f"human_grasped_tool: {result.get('human_grasped_tool', '')}", (20, 62), _state_color(state_str), scale=0.6)
    _draw_text(frame, f"grasp_counter: {result.get('grasp_counter', 0)}", (20, 92), scale=0.6)
    _draw_text(frame, f"pinch_distance: {_fmt(result.get('pinch_distance'))}", (20, 122), scale=0.6)
    _draw_text(frame, f"palm_to_tool: {_fmt(result.get('palm_to_tool_distance'))}", (20, 152), scale=0.6)
    _draw_text(frame, f"contact_count: {result.get('contact_count', 0)}", (20, 182), scale=0.6)
    _draw_text(frame, f"overlap: {_fmt(result.get('hand_tool_overlap_ratio'))}", (20, 212), scale=0.6)
    _draw_text(frame, f"grasp_score: {result.get('grasp_score', 0)}", (20, 242), scale=0.6)
    _draw_text(frame, f"mask_contact: {result.get('mask_contact_count', 0)}", (20, 272), scale=0.6)
    _draw_text(frame, f"mask_state: {mask_state.get('state', '')}", (20, 302), scale=0.6)
    _draw_text(frame, f"depth_contact: {result.get('depth_contact_count', 0)}", (20, 332), scale=0.6)
    _draw_text(frame, f"depth_diff: {_fmt(result.get('min_hand_tool_depth_diff_mm'))}mm", (20, 362), scale=0.6)
    _draw_text(frame, f"hands: {len(hand_infos)}", (20, 392), scale=0.6)
    _draw_text(frame, f"tool_source: {tool_source}", (20, 422), scale=0.6)
    _draw_text(frame, f"infer_ms: {state.get('infer_ms', 0.0):.1f}", (20, 452), scale=0.6)
    _draw_text(frame, f"FPS: {fps:.1f}", (20, 482), scale=0.6)


# ---------------------------------------------------------------------------
# Async coroutines
# ---------------------------------------------------------------------------

class LatestFrame:
    """Single-slot buffer: producer overwrites, consumer reads latest."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jpg: Optional[bytes] = None
        self._frame: Optional[np.ndarray] = None
        self._seq: int = 0

    async def put(self, frame: np.ndarray, jpg: bytes) -> None:
        async with self._lock:
            self._frame = frame
            self._jpg = jpg
            self._seq += 1

    async def get(self) -> tuple:
        async with self._lock:
            return self._frame, self._jpg, self._seq


async def camera_loop(
    cap: cv2.VideoCapture,
    latest: LatestFrame,
    jpeg_quality: int,
    stop: asyncio.Event,
) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            await asyncio.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        ok2, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ok2:
            continue
        await latest.put(frame, buf.tobytes())
        await asyncio.sleep(0)


async def sender_loop(
    ws,
    latest: LatestFrame,
    cmd_queue: asyncio.Queue,
    send_fps: float,
    stop: asyncio.Event,
) -> None:
    """Send JPEG frames and control commands over the same WebSocket."""
    interval = 1.0 / max(send_fps, 0.1)
    last_seq = -1
    while not stop.is_set():
        # Drain control commands first (high priority, no dropping)
        while not cmd_queue.empty():
            try:
                cmd = cmd_queue.get_nowait()
                await ws.send(cmd)
            except asyncio.QueueEmpty:
                break
            except websockets.ConnectionClosed:
                stop.set()
                return

        # Send latest frame
        _, jpg, seq = await latest.get()
        if jpg is None or seq == last_seq:
            await asyncio.sleep(0.005)
            continue
        last_seq = seq
        try:
            await ws.send(jpg)
        except websockets.ConnectionClosed:
            stop.set()
            return
        await asyncio.sleep(interval)


async def receiver_loop(ws, state: dict, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            msg = await ws.recv()
        except websockets.ConnectionClosed:
            stop.set()
            return
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            continue
        state["result"] = payload.get("result")
        state["hand_infos"] = payload.get("hand_infos", [])
        state["active_hand_index"] = payload.get("active_hand_index")
        state["tool_roi"] = payload.get("tool_roi")
        state["tool_detection"] = payload.get("tool_detection")
        state["tool_source"] = payload.get("tool_source", "none")
        state["mask_state"] = payload.get("mask_state", {})
        state["mask"] = _decode_mask(payload.get("mask_b64"))
        state["infer_ms"] = payload.get("infer_ms", 0.0)
        state["last_recv"] = time.time()


async def render_loop(
    latest: LatestFrame,
    state: dict,
    cmd_queue: asyncio.Queue,
    stop: asyncio.Event,
) -> None:
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    selection_frame: Optional[np.ndarray] = None

    while not stop.is_set():
        frame, _, _ = await latest.get()
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        view = frame.copy()
        selection_frame = frame.copy()
        _draw_overlay(view, state, fps)
        cv2.imshow(WINDOW_NAME, view)

        fps_n += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_n / (time.time() - fps_t0)
            fps_n = 0
            fps_t0 = time.time()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            stop.set()
            break

        elif key == ord("r"):
            await cmd_queue.put(json.dumps({"cmd": "reset"}))
            print("[client] sent reset")

        elif key == ord("s"):
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = LOG_DIR / f"screenshot_{ts}.png"
            cv2.imwrite(str(path), view)
            print(f"[client] screenshot saved: {path}")

        elif key == ord("m"):
            if selection_frame is not None:
                # cv2.selectROI is blocking — event loop pauses during selection
                roi = cv2.selectROI(WINDOW_NAME, selection_frame, showCrosshair=True, fromCenter=False)
                x, y, w, h = (int(v) for v in roi)
                if w > 0 and h > 0:
                    await cmd_queue.put(json.dumps({"cmd": "set_roi", "roi": [x, y, x + w, y + h]}))
                    print(f"[client] sent set_roi: [{x},{y},{x+w},{y+h}]")
                else:
                    print("[client] ROI selection cancelled")

        elif key == ord("c"):
            await cmd_queue.put(json.dumps({"cmd": "clear_roi"}))
            print("[client] sent clear_roi")

        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    latest = LatestFrame()
    cmd_queue: asyncio.Queue = asyncio.Queue()
    state: dict = {
        "result": None,
        "hand_infos": [],
        "active_hand_index": None,
        "tool_roi": None,
        "tool_detection": None,
        "tool_source": "none",
        "mask_state": {},
        "mask": None,
        "infer_ms": 0.0,
        "last_recv": 0.0,
    }
    stop = asyncio.Event()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    backoff = 1.0
    try:
        while not stop.is_set():
            try:
                print(f"[client] connecting to {args.url} ...")
                async with websockets.connect(
                    args.url,
                    max_size=16 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    print("[client] connected.")
                    backoff = 1.0
                    tasks = [
                        asyncio.create_task(camera_loop(cap, latest, args.jpeg_quality, stop)),
                        asyncio.create_task(sender_loop(ws, latest, cmd_queue, args.send_fps, stop)),
                        asyncio.create_task(receiver_loop(ws, state, stop)),
                        asyncio.create_task(render_loop(latest, state, cmd_queue, stop)),
                    ]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()
                    for t in pending:
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
            except (OSError, websockets.InvalidURI, websockets.InvalidHandshake) as exc:
                print(f"[client] connection error: {exc!r}; retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
            except Exception as exc:
                print(f"[client] unexpected error: {exc!r}")
                stop.set()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local camera client for remote grasp detection")
    p.add_argument("--url", default="ws://220.149.82.132:8766/infer", help="WebSocket URL, e.g. ws://server-host:8765/infer")
    p.add_argument("--cam", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (lower = less bandwidth)")
    p.add_argument("--send-fps", type=float, default=15.0, help="Max frames per second to send to server")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[client] interrupted")


if __name__ == "__main__":
    main()
