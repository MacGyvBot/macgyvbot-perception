"""
YOLO + MobileSAM + GraspNet client (RealSense → server → render).

Protocol:
  1. Sends intrinsics JSON once (text)
  2. Sends binary frames: [4B jpeg_len][JPEG][depth PNG] at --send-fps
  3. Renders received dets (masks/boxes) + grasp poses (projected to 2D)

Run:
    python remote_infer/graspnet_client.py \
        --url ws://<server-ip>:8767/infer \
        --send-fps 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from typing import Optional

import cv2
import numpy as np
import websockets

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ── Visualization ──────────────────────────────────────────────────────────
DET_COLORS = [
    (0,   200, 255), (0,   255, 100), (255, 100, 200), (100, 100, 255),
    (255, 255,   0), (255, 128,   0), (0,   180, 180), (180,   0, 180),
]
GRASP_PALETTE = [
    (0, 255,   0), (0, 220,  80), (0, 180, 180), (0, 140, 240), (0, 100, 255),
]
MASK_ALPHA = 0.45


def _det_color(cls: int) -> tuple[int, int, int]:
    return DET_COLORS[cls % len(DET_COLORS)]


def _grasp_color(rank: int) -> tuple[int, int, int]:
    return GRASP_PALETTE[min(rank, len(GRASP_PALETTE) - 1)]


def _project(xyz: list[float], fx: float, fy: float,
             cx: float, cy: float) -> tuple[int, int] | None:
    z = xyz[2]
    if z <= 0.01:
        return None
    return int(fx * xyz[0] / z + cx), int(fy * xyz[1] / z + cy)


def draw_results(
    frame: np.ndarray,
    dets: list[dict],
    grasps: list[dict],
    fx: float, fy: float, cx: float, cy: float,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Filled masks
    for d in dets:
        mask_norm = d.get("mask")
        if not mask_norm:
            continue
        pts = np.array([[int(nx * w), int(ny * h)] for nx, ny in mask_norm],
                       dtype=np.int32)
        cv2.fillPoly(overlay, [pts], _det_color(int(d.get("cls", 0))))
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, frame)

    # Contours + boxes + labels
    for d in dets:
        color = _det_color(int(d.get("cls", 0)))
        label = f"{d.get('name', '?')} {d.get('conf', 0.0):.2f}"
        mask_norm = d.get("mask")
        if mask_norm:
            pts = np.array([[int(nx * w), int(ny * h)] for nx, ny in mask_norm],
                           dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        if "xyxy" in d:
            x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Grasps projected to 2D
    for rank, g in enumerate(grasps):
        color = _grasp_color(rank)
        t   = g["translation"]       # [x, y, z]
        R   = np.array(g["rotation"])  # (3, 3)
        wid = g["width"]
        score = g["score"]

        center = _project(t, fx, fy, cx, cy)
        if center is None or not (0 <= center[0] < w and 0 <= center[1] < h):
            continue
        u, v = center

        radius = max(5, int(18 / max(t[2], 0.1)))
        cv2.circle(frame, (u, v), radius, color, 2)
        cv2.circle(frame, (u, v), 3, color, -1)

        # approach arrow  (-R[:,0] is approach direction in GraspNet convention)
        approach = [-R[i][0] for i in range(3)]
        tip = [t[i] + approach[i] * 0.05 for i in range(3)]
        tip_2d = _project(tip, fx, fy, cx, cy)
        if tip_2d is not None:
            cv2.arrowedLine(frame, (u, v), tip_2d, color, 2, tipLength=0.35)

        # finger span
        half_w = wid / 2.0
        lp = [t[i] - R[i][1] * half_w for i in range(3)]
        rp = [t[i] + R[i][1] * half_w for i in range(3)]
        l2d = _project(lp, fx, fy, cx, cy)
        r2d = _project(rp, fx, fy, cx, cy)
        if l2d and r2d:
            cv2.line(frame, l2d, r2d, color, 2)

        cv2.putText(frame, f"#{rank+1} {score:.2f}", (u + radius + 3, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# ── Async plumbing ─────────────────────────────────────────────────────────

class LatestRGBD:
    """Single-slot RGB-D buffer."""
    def __init__(self) -> None:
        self._lock  = asyncio.Lock()
        self._frame: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._jpg:   Optional[bytes]      = None
        self._seq:   int = 0

    async def put(self, frame: np.ndarray, depth: np.ndarray, jpg: bytes) -> None:
        async with self._lock:
            self._frame = frame
            self._depth = depth
            self._jpg   = jpg
            self._seq  += 1

    async def get(self):
        async with self._lock:
            return self._frame, self._depth, self._jpg, self._seq


async def realsense_loop(
    pipeline: "rs.pipeline",
    align: "rs.align",
    depth_scale: float,
    latest: LatestRGBD,
    jpeg_quality: int,
    stop: asyncio.Event,
) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    while not stop.is_set():
        frames  = pipeline.wait_for_frames(timeout_ms=3000)
        aligned = align.process(frames)
        depth_f = aligned.get_depth_frame()
        color_f = aligned.get_color_frame()
        if not depth_f or not color_f:
            await asyncio.sleep(0.005)
            continue
        frame = np.asanyarray(color_f.get_data())   # BGR uint8
        depth = np.asanyarray(depth_f.get_data())   # uint16
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            continue
        await latest.put(frame, depth, buf.tobytes())
        await asyncio.sleep(0)


async def sender_loop(
    ws,
    latest: LatestRGBD,
    send_fps: float,
    stop: asyncio.Event,
) -> None:
    interval = 1.0 / max(send_fps, 0.1)
    last_seq = -1
    while not stop.is_set():
        frame, depth, jpg, seq = await latest.get()
        if jpg is None or seq == last_seq:
            await asyncio.sleep(0.005)
            continue
        last_seq = seq

        # Encode depth as 16-bit PNG
        ok, depth_buf = cv2.imencode(".png", depth)
        if not ok:
            continue
        depth_bytes = depth_buf.tobytes()

        # Pack: [4B jpeg_len][jpeg][depth_png]
        header = struct.pack(">I", len(jpg))
        try:
            await ws.send(header + jpg + depth_bytes)
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
        state.update({
            "dets":     payload.get("dets",     []),
            "grasps":   payload.get("grasps",   []),
            "yolo_ms":  payload.get("yolo_ms",  0.0),
            "sam_ms":   payload.get("sam_ms",   0.0),
            "grasp_ms": payload.get("grasp_ms", 0.0),
            "total_ms": payload.get("total_ms", 0.0),
            "last_recv": time.time(),
        })


async def render_loop(
    latest: LatestRGBD,
    state: dict,
    fx: float, fy: float, cx: float, cy: float,
    stop: asyncio.Event,
    window: str,
) -> None:
    fps_t0, fps_n, fps = time.time(), 0, 0.0
    while not stop.is_set():
        frame, _, _, _ = await latest.get()
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        view = frame.copy()
        draw_results(view, state.get("dets", []), state.get("grasps", []),
                     fx, fy, cx, cy)

        info = (
            f"FPS {fps:5.1f}  "
            f"YOLO {state.get('yolo_ms',0):.0f}ms  "
            f"SAM {state.get('sam_ms',0):.0f}ms  "
            f"GraspNet {state.get('grasp_ms',0):.0f}ms  "
            f"dets {len(state.get('dets',[]))}  "
            f"grasps {len(state.get('grasps',[]))}"
        )
        for th, col in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
            cv2.putText(view, info, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, th, cv2.LINE_AA)

        cv2.imshow(window, view)
        fps_n += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps    = fps_n / (now - fps_t0)
            fps_n  = 0
            fps_t0 = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop.set()
            break
        await asyncio.sleep(0)


# ── Main ───────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    if not REALSENSE_AVAILABLE:
        raise RuntimeError("pyrealsense2 not found — pip install pyrealsense2")

    print("[client] initializing RealSense...")
    pipeline = rs.pipeline()
    cfg_rs   = rs.config()
    cfg_rs.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,  30)
    cfg_rs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    profile = pipeline.start(cfg_rs)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr  = (profile.get_stream(rs.stream.color)
                    .as_video_stream_profile().get_intrinsics())
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(f"[client] intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"[client] depth_scale={depth_scale:.6f}")

    intrinsics_msg = json.dumps({
        "type": "intrinsics",
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "depth_scale": depth_scale,
    })

    latest = LatestRGBD()
    state: dict = {
        "dets": [], "grasps": [],
        "yolo_ms": 0.0, "sam_ms": 0.0, "grasp_ms": 0.0, "total_ms": 0.0,
        "last_recv": 0.0,
    }
    stop   = asyncio.Event()
    window = "YOLO + MobileSAM + GraspNet (remote)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    backoff = 1.0
    try:
        while not stop.is_set():
            try:
                print(f"[client] connecting to {args.url}")
                async with websockets.connect(
                    args.url,
                    max_size=32 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    print("[client] connected — sending intrinsics")
                    await ws.send(intrinsics_msg)
                    backoff = 1.0

                    tasks = [
                        asyncio.create_task(realsense_loop(
                            pipeline, align, depth_scale, latest,
                            args.jpeg_quality, stop)),
                        asyncio.create_task(sender_loop(
                            ws, latest, args.send_fps, stop)),
                        asyncio.create_task(receiver_loop(ws, state, stop)),
                        asyncio.create_task(render_loop(
                            latest, state, fx, fy, cx, cy, stop, window)),
                    ]
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()
                    for t in pending:
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

            except (OSError, websockets.InvalidURI,
                    websockets.InvalidHandshake) as e:
                print(f"[client] connection error: {e!r}; retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
            except Exception as e:
                print(f"[client] unexpected error: {e!r}")
                stop.set()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GraspNet RealSense client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",           required=True,
                   help="WebSocket URL, e.g. ws://server-ip:8767/infer")
    p.add_argument("--width",         type=int,   default=640)
    p.add_argument("--height",        type=int,   default=480)
    p.add_argument("--send-fps",      type=float, default=8.0,
                   help="Frame send rate (match to server throughput ~5-10 Hz)")
    p.add_argument("--jpeg-quality",  type=int,   default=80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[client] interrupted")


if __name__ == "__main__":
    main()
