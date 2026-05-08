"""
YOLO + MobileSAM + GraspNet-baseline pipeline (Intel RealSense RGB-D).

Pipeline:
    RealSense RGB-D → YOLO bbox → MobileSAM mask
        → masked depth → point cloud → GraspNet → grasp visualization

Requirements:
    pip install pyrealsense2
    Build CUDA extensions once:
        cd graspnet_baseline/pointnet2 && python setup.py build_ext --inplace
        cd graspnet_baseline/knn       && python setup.py build_ext --inplace
    Checkpoint: graspnet_baseline/logs/checkpoint-rs.tar
        (download: https://github.com/graspnet/graspnet-baseline)

Run:
    python yolo_mobilesam_graspnet.py \
        --yolo  runs/detect/.../best.pt \
        --sam   mobile_sam.pt \
        --ckpt  graspnet_baseline/logs/checkpoint-rs.tar \
        --graspnet-path graspnet_baseline \
        --device cuda

Real-time note:
    YOLO+SAM display loop : ~20-25 FPS  (main thread)
    GraspNet updates      : ~5-10 Hz    (background worker thread)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ── Visualization constants ────────────────────────────────────────────────
DET_COLORS = [
    (0,   200, 255), (0,   255, 100), (255, 100, 200), (100, 100, 255),
    (255, 255,   0), (255, 128,   0), (0,   180, 180), (180,   0, 180),
]
GRASP_PALETTE = [
    (0, 255,   0), (0, 220,  80), (0, 180, 180), (0, 140, 240), (0, 100, 255),
]
MASK_ALPHA = 0.40


def _det_color(cls: int) -> tuple[int, int, int]:
    return DET_COLORS[cls % len(DET_COLORS)]


def _grasp_color(rank: int) -> tuple[int, int, int]:
    return GRASP_PALETTE[min(rank, len(GRASP_PALETTE) - 1)]


# ── Geometry helpers ───────────────────────────────────────────────────────

def mask_to_polygon(binary_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    pts = max(contours, key=cv2.contourArea).squeeze(1)
    return pts if pts.ndim == 2 else None


def masked_depth_to_pointcloud(
    depth_img: np.ndarray,
    color_img: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
    z_min: float = 0.20, z_max: float = 1.50,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    z = depth_img[rows, cols].astype(np.float32) * depth_scale
    valid = (z > z_min) & (z < z_max)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if len(z) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = color_img[rows, cols].astype(np.float32) / 255.0
    return points, colors


def project_3d_to_2d(
    xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int] | None:
    if xyz[2] <= 0.01:
        return None
    return int(fx * xyz[0] / xyz[2] + cx), int(fy * xyz[1] / xyz[2] + cy)


# ── Visualization ─────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, dets: list[dict]) -> None:
    overlay = frame.copy()
    for d in dets:
        pts = d.get("polygon")
        if pts is not None:
            cv2.fillPoly(overlay, [pts], _det_color(d["cls"]))
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, frame)
    for d in dets:
        color = _det_color(d["cls"])
        label = f"{d['name']} {d['conf']:.2f}"
        pts = d.get("polygon")
        if pts is not None:
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(10, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


_FINGER_DEPTH = 0.04   # m — length of each finger segment


def draw_grasps(
    frame: np.ndarray, grasps,
    fx: float, fy: float, cx: float, cy: float,
    top_k: int = 5, img_h: int = 480, img_w: int = 640,
) -> None:
    """Draw gripper shapes: palm bar + two fingers + approach arrow.

    GraspNet convention:
        R[:,0] — axis pointing AWAY from object (opposite of approach)
        R[:,1] — finger-spread (width) axis
        t      — grasp centre, between the two fingertips
    """
    if grasps is None or len(grasps) == 0:
        return
    n = min(top_k, len(grasps))
    for rank in range(n):
        t      = grasps.translations[rank]
        R      = grasps.rotation_matrices[rank]
        score  = grasps.scores[rank]
        half_w = grasps.widths[rank] / 2.0
        color  = _grasp_color(rank)

        # 3-D key points
        tip_L   = t - R[:, 1] * half_w                    # left  fingertip
        tip_R   = t + R[:, 1] * half_w                    # right fingertip
        base_L  = tip_L + R[:, 0] * _FINGER_DEPTH         # left  finger base (palm side)
        base_R  = tip_R + R[:, 0] * _FINGER_DEPTH         # right finger base
        palm_c  = (base_L + base_R) / 2                   # palm centre (arrow tail)

        # Project all to 2-D
        p_tL = project_3d_to_2d(tip_L,  fx, fy, cx, cy)
        p_tR = project_3d_to_2d(tip_R,  fx, fy, cx, cy)
        p_bL = project_3d_to_2d(base_L, fx, fy, cx, cy)
        p_bR = project_3d_to_2d(base_R, fx, fy, cx, cy)
        p_ct = project_3d_to_2d(t,      fx, fy, cx, cy)
        p_pc = project_3d_to_2d(palm_c, fx, fy, cx, cy)

        if any(p is None for p in (p_tL, p_tR, p_bL, p_bR, p_ct)):
            continue
        if not (0 <= p_ct[0] < img_w and 0 <= p_ct[1] < img_h):
            continue

        # Gripper shape: palm ─ left finger ─ right finger
        cv2.line(frame, p_bL, p_bR, color, 3)   # palm bar
        cv2.line(frame, p_bL, p_tL, color, 3)   # left  finger
        cv2.line(frame, p_bR, p_tR, color, 3)   # right finger

        # Approach arrow: palm centre → grasp centre (toward object)
        if p_pc is not None:
            cv2.arrowedLine(frame, p_pc, p_ct, color, 2, tipLength=0.3)

        # Label above the palm bar
        label = f"#{rank+1} s={score:.2f} z={t[2]:.2f}m"
        lx = min(p_bL[0], p_bR[0])
        ly = min(p_bL[1], p_bR[1]) - 6
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,    1, cv2.LINE_AA)


def _rmat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 rotation matrix → (roll, pitch, yaw) in degrees. ZYX convention."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:  # gimbal lock
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


def _print_grasps(grasps, call_idx: int, top_k: int) -> None:
    if grasps is None or len(grasps) == 0:
        return
    n = min(top_k, len(grasps))
    print(f"\n[grasp #{call_idx}]  top-{n}  (camera frame | position: m, angle: deg)")
    print(f"  {'#':>3}  {'score':>6}  {'width':>6}"
          f"  {'x':>8}  {'y':>8}  {'z':>8}"
          f"  {'roll':>8}  {'pitch':>8}  {'yaw':>8}")
    for i in range(n):
        t = grasps.translations[i]
        r, p, y = _rmat_to_rpy(grasps.rotation_matrices[i])
        print(f"  {i+1:>3}  {grasps.scores[i]:>6.3f}  {grasps.widths[i]:>6.3f}"
              f"  {t[0]:>8.4f}  {t[1]:>8.4f}  {t[2]:>8.4f}"
              f"  {r:>8.2f}  {p:>8.2f}  {y:>8.2f}")


# ── Minimal GraspGroup ─────────────────────────────────────────────────────

class GraspGroup:
    """
    Wraps the (N, 17) array from GraspNet pred_decode.
    Layout: [score, width, height, depth, rotation×9, center×3, obj_id]
    """
    _S = 0
    _W = 1
    _R = slice(4, 13)
    _C = slice(13, 16)

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
        order = np.argsort(-self.scores)
        data  = self._d[order]
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


# ── GraspNet loader & wrapper ──────────────────────────────────────────────

class GraspNetWrapper:
    _NUM_POINT = 20_000

    def __init__(self, net, device: str) -> None:
        self._net    = net
        self._device = device

    def get_grasp(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        lims: list[float],
    ) -> tuple["GraspGroup | None", None]:
        from models.graspnet import pred_decode  # noqa: PLC0415

        xmin, xmax, ymin, ymax, zmin, zmax = lims
        mask = (
            (points[:, 0] > xmin) & (points[:, 0] < xmax) &
            (points[:, 1] > ymin) & (points[:, 1] < ymax) &
            (points[:, 2] > zmin) & (points[:, 2] < zmax)
        )
        pts = points[mask].astype(np.float32)
        if len(pts) < 50:
            return None, None

        N   = len(pts)
        idx = np.random.choice(N, self._NUM_POINT, replace=(N < self._NUM_POINT))
        pts_t = torch.FloatTensor(pts[idx]).unsqueeze(0).to(self._device)

        with torch.no_grad():
            ep    = self._net({"point_clouds": pts_t})
            preds = pred_decode(ep)

        arr = preds[0].detach().cpu().numpy()
        if arr.shape[0] == 0:
            return None, None
        return GraspGroup(arr), None


def load_graspnet(args: argparse.Namespace) -> GraspNetWrapper | None:
    graspnet_path = args.graspnet_path
    for subdir in ["", "models", "utils", "pointnet2", "knn"]:
        p = graspnet_path if not subdir else f"{graspnet_path}/{subdir}"
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from models.graspnet import GraspNet  # noqa: PLC0415
    except ImportError as e:
        print(f"[warn] GraspNet import failed: {e}")
        return None

    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)

    print(f"[info] loading GraspNet checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(state.get("model_state_dict", state))
    net.to(args.device).eval()
    print("[info] GraspNet ready")
    return GraspNetWrapper(net, args.device)


# ── Background grasp worker ────────────────────────────────────────────────

class GraspWorker:
    """Runs GraspNet in a daemon thread so the display loop never blocks."""

    def __init__(self, model: GraspNetWrapper, lims: list[float]) -> None:
        self._model   = model
        self._lims    = lims
        self._lock    = threading.Lock()
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._result  = None
        self._ms: float = 0.0
        self._n: int    = 0
        threading.Thread(target=self._loop, name="grasp-worker", daemon=True).start()

    def submit(self, points: np.ndarray, colors: np.ndarray) -> None:
        with self._lock:
            self._pending = (points, colors)

    def get_result(self) -> tuple:
        with self._lock:
            return self._result, self._ms, self._n

    def _loop(self) -> None:
        while True:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None or len(job[0]) < 50:
                time.sleep(0.01)
                continue
            t0 = time.time()
            try:
                gg, _ = self._model.get_grasp(job[0], job[1], lims=self._lims)
                ms = (time.time() - t0) * 1000.0
                if gg is not None and len(gg) > 0:
                    gg = gg.nms().sort_by_score()
                with self._lock:
                    self._result = gg
                    self._ms = ms
                    self._n += 1
            except Exception as exc:
                print(f"[grasp-worker] {exc}")
            time.sleep(0.005)


# ── Main pipeline ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable — falling back to CPU")
        device = "cpu"

    print(f"[info] loading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)
    yolo.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8),
                 device=device, imgsz=args.imgsz, verbose=False)
    print("[info] YOLO ready")

    print(f"[info] loading MobileSAM: {args.sam}")
    sam_model = sam_model_registry["vit_t"](checkpoint=args.sam)
    sam_model.to(device=device).eval()
    predictor = SamPredictor(sam_model)
    print("[info] MobileSAM ready")

    grasp_model = load_graspnet(args)
    grasp_worker: GraspWorker | None = None
    if grasp_model is not None:
        lims = [args.lim_xmin, args.lim_xmax, args.lim_ymin,
                args.lim_ymax, args.lim_zmin, args.lim_zmax]
        grasp_worker = GraspWorker(grasp_model, lims)
        print(f"[info] GraspNet worker started  workspace={lims}")

    if not REALSENSE_AVAILABLE:
        raise RuntimeError("pyrealsense2 not found — pip install pyrealsense2")

    print("[info] initializing RealSense...")
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
    print(f"[info] intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print("[info] running — press 'q' to quit")

    window = "YOLO + MobileSAM + GraspNet"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    fps_t0, fps_n, fps = time.time(), 0, 0.0
    _last_grasp_n = -1  # track when worker produces a new result

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f:
                continue

            depth_img = np.asanyarray(depth_f.get_data())
            frame     = np.asanyarray(color_f.get_data())
            h, w      = frame.shape[:2]

            # YOLO
            t0 = time.time()
            results = yolo.predict(frame, device=device, imgsz=args.imgsz,
                                   conf=args.conf, iou=args.iou, verbose=False)
            yolo_ms = (time.time() - t0) * 1000.0

            r     = results[0]
            boxes = r.boxes
            dets: list[dict] = []
            sam_ms       = 0.0
            combined_mask = np.zeros((h, w), dtype=np.uint8)

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
                    dets.append({
                        "xyxy": box.tolist(), "cls": int(c),
                        "name": str(names[int(c)]), "conf": float(p),
                        "polygon": mask_to_polygon(bin_mask),
                    })
                sam_ms = (time.time() - t0) * 1000.0

            # Submit point cloud to GraspNet worker
            if grasp_worker is not None and combined_mask.any():
                pts, cols = masked_depth_to_pointcloud(
                    depth_img, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    combined_mask, fx, fy, cx, cy,
                    depth_scale=depth_scale,
                    z_min=args.lim_zmin, z_max=args.lim_zmax,
                )
                if len(pts) >= 50:
                    grasp_worker.submit(pts, cols)

            draw_detections(frame, dets)

            if grasp_worker is not None:
                grasps, grasp_ms, n_calls = grasp_worker.get_result()
                draw_grasps(frame, grasps, fx, fy, cx, cy,
                            top_k=args.top_k, img_h=h, img_w=w)
                grasp_hud = f"GraspNet {grasp_ms:.0f}ms #{n_calls}"
                if n_calls != _last_grasp_n:
                    _print_grasps(grasps, n_calls, args.top_k)
                    _last_grasp_n = n_calls
            else:
                grasp_hud = "GraspNet: OFF"

            info = (f"FPS {fps:5.1f}  YOLO {yolo_ms:.0f}ms  "
                    f"SAM {sam_ms:.0f}ms  {grasp_hud}  dets {len(dets)}")
            for th, col in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
                cv2.putText(frame, info, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, th, cv2.LINE_AA)

            cv2.imshow(window, frame)
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps    = fps_n / (now - fps_t0)
                fps_n  = 0
                fps_t0 = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO + MobileSAM + GraspNet (Intel RealSense)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--yolo",          default="runs/detect/train/weights/best.pt")
    p.add_argument("--sam",           default="mobile_sam.pt")
    p.add_argument("--ckpt",          default="graspnet_baseline/logs/checkpoint-rs.tar")
    p.add_argument("--graspnet-path", default="graspnet_baseline", dest="graspnet_path")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--width",  type=int,   default=640)
    p.add_argument("--height", type=int,   default=480)
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
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
