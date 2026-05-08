"""
YOLO + MobileSAM + GraspNet inference server (FastAPI + WebSocket).

Protocol (per connection):
  1. Client → Server  [text]   intrinsics JSON once:
         {"type":"intrinsics","fx":…,"fy":…,"cx":…,"cy":…,"depth_scale":…}
  2. Client → Server  [binary] each frame:
         [4 bytes: jpeg_len (big-endian uint32)][JPEG bytes][depth PNG bytes]
  3. Server → Client  [text]   JSON result:
         {
           "yolo_ms": float, "sam_ms": float, "grasp_ms": float, "total_ms": float,
           "dets": [{"xyxy":[…],"cls":int,"name":str,"conf":float,
                     "mask":[[nx,ny],…] | null}, …],
           "grasps": [{"score":float,"width":float,
                       "translation":[x,y,z],"rotation":[[…],[…],[…]]}, …]
         }

Run:
    python remote_infer/graspnet_server.py \
        --yolo  runs/detect/.../best.pt \
        --sam   mobile_sam.pt \
        --ckpt  graspnet_baseline/logs/checkpoint-rs.tar \
        --graspnet-path graspnet_baseline \
        --device 0 --port 8767
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mobile_sam import SamPredictor, sam_model_registry
from ultralytics import YOLO


# ── GraspNet helpers (mirrored from standalone script) ─────────────────────

class GraspGroup:
    _S = 0; _W = 1; _R = slice(4, 13); _C = slice(13, 16)

    def __init__(self, data: np.ndarray) -> None:
        self._d = data.astype(np.float32)

    def __len__(self) -> int:
        return len(self._d)

    @property
    def scores(self)           -> np.ndarray: return self._d[:, self._S]
    @property
    def widths(self)           -> np.ndarray: return self._d[:, self._W]
    @property
    def translations(self)     -> np.ndarray: return self._d[:, self._C]
    @property
    def rotation_matrices(self)-> np.ndarray: return self._d[:, self._R].reshape(-1, 3, 3)

    def sort_by_score(self) -> "GraspGroup":
        return GraspGroup(self._d[np.argsort(-self.scores)])

    def nms(self, t_thresh: float = 0.03, r_thresh: float = 30.0) -> "GraspGroup":
        if len(self._d) == 0:
            return GraspGroup(self._d)
        order   = np.argsort(-self.scores)
        data    = self._d[order]
        centers = data[:, self._C]
        rots    = data[:, self._R].reshape(-1, 3, 3)
        keep    = np.ones(len(data), dtype=bool)
        for i in range(len(data)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(data)):
                if not keep[j]:
                    continue
                if np.linalg.norm(centers[i] - centers[j]) > t_thresh:
                    continue
                tr = np.clip((np.trace(rots[i].T @ rots[j]) - 1.0) / 2.0, -1.0, 1.0)
                if np.degrees(np.arccos(tr)) < r_thresh:
                    keep[j] = False
        return GraspGroup(data[keep])


def masked_depth_to_pointcloud(
    depth_img: np.ndarray, mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float, z_min: float, z_max: float,
) -> np.ndarray:
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return np.empty((0, 3), np.float32)
    z = depth_img[rows, cols].astype(np.float32) * depth_scale
    valid = (z > z_min) & (z < z_max)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if len(z) == 0:
        return np.empty((0, 3), np.float32)
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _rmat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


def _print_grasps(grasps: list[dict], peer: str) -> None:
    print(f"\n[server grasp | {peer}]  {len(grasps)} grasps"
          f"  (camera frame | position: m, angle: deg)")
    print(f"  {'#':>3}  {'score':>6}  {'width':>6}"
          f"  {'x':>8}  {'y':>8}  {'z':>8}"
          f"  {'roll':>8}  {'pitch':>8}  {'yaw':>8}")
    for i, g in enumerate(grasps):
        t = g["translation"]
        R = np.array(g["rotation"])
        r, p, y = _rmat_to_rpy(R)
        print(f"  {i+1:>3}  {g['score']:>6.3f}  {g['width']:>6.3f}"
              f"  {t[0]:>8.4f}  {t[1]:>8.4f}  {t[2]:>8.4f}"
              f"  {r:>8.2f}  {p:>8.2f}  {y:>8.2f}")


def mask_to_norm_polygon(
    binary_mask: np.ndarray, fw: int, fh: int,
) -> list[list[float]] | None:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    pts = max(contours, key=cv2.contourArea).squeeze(1)
    if pts.ndim != 2:
        return None
    return [[float(x / fw), float(y / fh)] for x, y in pts]


# ── GraspNet loader ────────────────────────────────────────────────────────

def load_graspnet(graspnet_path: str, ckpt: str, device: str):
    for subdir in ["", "models", "utils", "pointnet2", "knn"]:
        p = graspnet_path if not subdir else f"{graspnet_path}/{subdir}"
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.graspnet import GraspNet  # noqa: PLC0415
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    state = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(state.get("model_state_dict", state))
    net.to(device).eval()
    print(f"[server] GraspNet ready  ckpt={ckpt}")
    return net


# ── FastAPI app builder ────────────────────────────────────────────────────

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="graspnet-inference-server")

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"

    # YOLO
    print(f"[server] loading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)
    yolo.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8),
                 device=device, imgsz=args.imgsz, verbose=False)
    print("[server] YOLO ready")

    # MobileSAM
    print(f"[server] loading MobileSAM: {args.sam}")
    sam_model = sam_model_registry["vit_t"](checkpoint=args.sam)
    sam_model.to(device=device).eval()
    predictor = SamPredictor(sam_model)
    print("[server] MobileSAM ready")

    # GraspNet
    graspnet = load_graspnet(args.graspnet_path, args.ckpt, device)

    _NUM_POINT = 20_000
    lims = [args.lim_xmin, args.lim_xmax,
            args.lim_ymin, args.lim_ymax,
            args.lim_zmin, args.lim_zmax]

    def _infer_sync(
        frame: np.ndarray,
        depth_img: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        depth_scale: float,
    ) -> dict[str, Any]:
        fh, fw = frame.shape[:2]
        out: dict[str, Any] = {
            "yolo_ms": 0.0, "sam_ms": 0.0, "grasp_ms": 0.0, "total_ms": 0.0,
            "dets": [], "grasps": [],
        }
        t_total = time.time()

        # YOLO
        t0 = time.time()
        results = yolo.predict(frame, device=device, imgsz=args.imgsz,
                               conf=args.conf, iou=args.iou, verbose=False)
        out["yolo_ms"] = (time.time() - t0) * 1000.0

        r     = results[0]
        boxes = r.boxes
        combined_mask = np.zeros((fh, fw), dtype=np.uint8)

        if boxes is not None and len(boxes) > 0:
            xyxy  = boxes.xyxy.cpu().numpy()
            cls   = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            names = r.names
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            t0 = time.time()
            for box, c, p in zip(xyxy, cls, confs):
                masks, _, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=box[None, :], multimask_output=False,
                )
                bin_mask = masks[0].astype(np.uint8)
                combined_mask = np.maximum(combined_mask, bin_mask)
                out["dets"].append({
                    "xyxy": [float(v) for v in box],
                    "cls":  int(c),
                    "name": str(names[int(c)]),
                    "conf": float(p),
                    "mask": mask_to_norm_polygon(bin_mask, fw, fh),
                })
            out["sam_ms"] = (time.time() - t0) * 1000.0

        # GraspNet
        if combined_mask.any():
            pts = masked_depth_to_pointcloud(
                depth_img, combined_mask, fx, fy, cx, cy,
                depth_scale, lims[4], lims[5],
            )
            xmin, xmax, ymin, ymax = lims[:4]
            if len(pts) >= 50:
                t0  = time.time()
                ws_mask = (
                    (pts[:, 0] > xmin) & (pts[:, 0] < xmax) &
                    (pts[:, 1] > ymin) & (pts[:, 1] < ymax)
                )
                pts_ws = pts[ws_mask]
                if len(pts_ws) >= 50:
                    N   = len(pts_ws)
                    idx = np.random.choice(N, _NUM_POINT, replace=(N < _NUM_POINT))
                    pts_t = torch.FloatTensor(pts_ws[idx]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        from models.graspnet import pred_decode  # noqa: PLC0415
                        ep    = graspnet({"point_clouds": pts_t})
                        preds = pred_decode(ep)
                    arr = preds[0].detach().cpu().numpy()
                    if arr.shape[0] > 0:
                        gg = GraspGroup(arr).nms().sort_by_score()
                        top = min(args.top_k, len(gg))
                        out["grasps"] = [
                            {
                                "score":       float(gg.scores[i]),
                                "width":       float(gg.widths[i]),
                                "translation": gg.translations[i].tolist(),
                                "rotation":    gg.rotation_matrices[i].tolist(),
                            }
                            for i in range(top)
                        ]
                    out["grasp_ms"] = (time.time() - t0) * 1000.0

        out["total_ms"] = (time.time() - t_total) * 1000.0
        return out

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "device": device}

    @app.websocket("/infer")
    async def infer_ws(ws: WebSocket) -> None:
        await ws.accept()
        peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        print(f"[server] client connected: {peer}")
        loop = asyncio.get_running_loop()

        # Wait for intrinsics handshake
        try:
            raw = await ws.receive_text()
            intr = json.loads(raw)
            assert intr.get("type") == "intrinsics"
            fx          = float(intr["fx"])
            fy          = float(intr["fy"])
            cx          = float(intr["cx"])
            cy          = float(intr["cy"])
            depth_scale = float(intr["depth_scale"])
            print(f"[server] {peer} intrinsics: fx={fx:.1f} fy={fy:.1f} "
                  f"cx={cx:.1f} cy={cy:.1f} ds={depth_scale:.6f}")
        except Exception as e:
            print(f"[server] intrinsics handshake failed: {e!r}")
            await ws.close()
            return

        n = 0
        try:
            while True:
                data = await ws.receive_bytes()
                # parse [4B jpeg_len][jpeg][depth_png]
                if len(data) < 4:
                    continue
                jpeg_len  = struct.unpack(">I", data[:4])[0]
                jpeg_data = data[4:4 + jpeg_len]
                depth_data = data[4 + jpeg_len:]

                frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                depth = cv2.imdecode(np.frombuffer(depth_data, np.uint8), cv2.IMREAD_UNCHANGED)
                if frame is None or depth is None:
                    await ws.send_text(json.dumps({"error": "decode_failed"}))
                    continue

                payload = await loop.run_in_executor(
                    None, _infer_sync, frame, depth, fx, fy, cx, cy, depth_scale
                )
                await ws.send_text(json.dumps(payload))
                n += 1
                if payload["grasps"]:
                    _print_grasps(payload["grasps"], peer)
                if n % 20 == 0:
                    print(f"[server] {peer}  n={n}  "
                          f"yolo={payload['yolo_ms']:.0f}ms  "
                          f"sam={payload['sam_ms']:.0f}ms  "
                          f"grasp={payload['grasp_ms']:.0f}ms  "
                          f"total={payload['total_ms']:.0f}ms  "
                          f"grasps={len(payload['grasps'])}")
        except WebSocketDisconnect:
            print(f"[server] {peer} disconnected (n={n})")
        except Exception as e:
            print(f"[server] error on {peer}: {e!r}")
            try:
                await ws.close()
            except Exception:
                pass

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO + MobileSAM + GraspNet WebSocket server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--yolo",          default="/home/gunwoo/macgyvbot-perception/hand_grasp_detection/yolo/ckeckpoint/yolo_v11/yolov11_best.pt")
    p.add_argument("--sam",           default="/home/gunwoo/macgyvbot-perception/hand_grasp_detection/yolo/mobile_sam.pt")
    p.add_argument("--ckpt",          default="/home/gunwoo/macgyvbot-perception/hand_grasp_detection/yolo/graspnet_baseline/logs/checkpoint-rs.tar")
    p.add_argument("--graspnet-path", default="/home/gunwoo/macgyvbot-perception/hand_grasp_detection/yolo/graspnet_baseline", dest="graspnet_path")
    p.add_argument("--device",        default="0")
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--top-k",  type=int,   default=5, dest="top_k")
    p.add_argument("--lim-xmin", type=float, default=-0.5)
    p.add_argument("--lim-xmax", type=float, default=0.5)
    p.add_argument("--lim-ymin", type=float, default=-0.5)
    p.add_argument("--lim-ymax", type=float, default=0.5)
    p.add_argument("--lim-zmin", type=float, default=0.2)
    p.add_argument("--lim-zmax", type=float, default=1.5)
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8767)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app  = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
