#!/usr/bin/env python3
"""
Run YOLO on RealSense RGB frames and estimate object coordinates from aligned depth.

The reported camera_xyz coordinates use the RealSense optical camera frame:
X points right in the image, Y points down in the image, and Z points forward
from the camera, in meters.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

LOCAL_CONFIG_DIR = Path(__file__).resolve().parent / ".runtime_config"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CONFIG_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CONFIG_DIR / "ultralytics"))
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(LOCAL_CONFIG_DIR / "ultralytics"))
(LOCAL_CONFIG_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(LOCAL_CONFIG_DIR / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_YOLO_MODEL = r"D:\code\weight\best.pt"
DEFAULT_CAMERA_Z = 1.50
DEFAULT_FPS = 30
YOLO_CLASS_NAMES = ["robot", "cube", "bottle", "table", "stool", "smallpallet", "bigpallet"]


def _model_class_name(model_names: dict[int, str] | list[str], cls_id: int) -> str:
    if isinstance(model_names, dict) and cls_id in model_names:
        return str(model_names[cls_id]).lower()
    if isinstance(model_names, list) and 0 <= cls_id < len(model_names):
        return str(model_names[cls_id]).lower()
    if 0 <= cls_id < len(YOLO_CLASS_NAMES):
        return YOLO_CLASS_NAMES[cls_id]
    return str(cls_id)


def _fmt_vec(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return str(np.round(np.asarray(value, dtype=float), 3).tolist())


def _load_camera_pose(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "camera_to_world" in data:
        pose = np.asarray(data["camera_to_world"], dtype=np.float64)
    elif "cameraViewTransform" in data:
        view_mat = np.asarray(data["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
        pose = np.linalg.inv(view_mat.T)
    else:
        raise ValueError("pose json must contain 'camera_to_world' or 'cameraViewTransform'")
    return pose.reshape(4, 4)


def _default_camera_pose(camera_z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.0, 0.0, float(camera_z)]
    return pose


def _camera_to_world(point_camera: np.ndarray | None, camera_pose: np.ndarray | None) -> np.ndarray | None:
    if point_camera is None or camera_pose is None:
        return None
    point_h = np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
    return (camera_pose @ point_h)[:3]


def _safe_bbox(
    xyxy: list[float],
    width: int,
    height: int,
    shrink: float,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    box_w = x2 - x1
    box_h = y2 - y1
    inset_x = box_w * float(shrink)
    inset_y = box_h * float(shrink)
    sx1 = int(round(max(0, min(width - 1, x1 + inset_x))))
    sy1 = int(round(max(0, min(height - 1, y1 + inset_y))))
    sx2 = int(round(max(0, min(width, x2 - inset_x))))
    sy2 = int(round(max(0, min(height, y2 - inset_y))))
    if sx2 - sx1 < min_size or sy2 - sy1 < min_size:
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        half = max(1, min_size // 2)
        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(width, cx + half + 1)
        sy2 = min(height, cy + half + 1)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    return sx1, sy1, sx2, sy2


class RealSenseRgbdCapture:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        serial: str,
        timeout_ms: int = 5000,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is not installed. Install it with: pip install pyrealsense2") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.timeout_ms = int(timeout_ms)
        self.depth_scale = 0.001
        self.color_intrinsics = None
        self.width = int(width)
        self.height = int(height)

        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        try:
            self.profile = self.pipeline.start(config)
        except Exception:
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color)
            config.enable_stream(rs.stream.depth)
            self.profile = self.pipeline.start(config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self._read_color_intrinsics()

    def _read_color_intrinsics(self) -> None:
        color_profile = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.color_intrinsics = intr
        self.width = int(intr.width)
        self.height = int(intr.height)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned incomplete RGB-D frames")
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        return color_bgr, depth_raw

    def deproject_pixel(self, pixel: tuple[float, float], depth_m: float) -> np.ndarray:
        if self.color_intrinsics is None:
            raise RuntimeError("Color intrinsics are not available")
        point = self.rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m))
        return np.asarray(point, dtype=np.float64)

    def release(self) -> None:
        self.pipeline.stop()


class RgbdYoloCoordinateDetector:
    def __init__(
        self,
        model_path: str,
        *,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        depth_min: float,
        depth_max: float,
        bbox_shrink: float,
        min_depth_pixels: int,
        camera_pose: np.ndarray | None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.device = device.strip()
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.bbox_shrink = float(bbox_shrink)
        self.min_depth_pixels = int(min_depth_pixels)
        self.camera_pose = camera_pose

    def detect(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
        capture: RealSenseRgbdCapture,
    ) -> list[dict[str, Any]]:
        predict_kwargs: dict[str, Any] = {
            "source": color_bgr,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)[0]
        model_names = getattr(self.model, "names", None) or {}
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = _model_class_name(model_names, cls_id)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u, v, w, h = box.xywh[0].tolist()
            xyxy = [float(x1), float(y1), float(x2), float(y2)]
            depth_info = self._estimate_depth_xyz(xyxy, depth_raw, capture)
            detections.append(
                {
                    "class": class_name,
                    "confidence": float(box.conf[0].item()),
                    "xyxy": xyxy,
                    "xywh": [float(u), float(v), float(w), float(h)],
                    **depth_info,
                }
            )
        return detections

    def _estimate_depth_xyz(
        self,
        xyxy: list[float],
        depth_raw: np.ndarray,
        capture: RealSenseRgbdCapture,
    ) -> dict[str, Any]:
        height, width = depth_raw.shape[:2]
        crop_box = _safe_bbox(xyxy, width, height, self.bbox_shrink, min_size=7)
        fallback_crop_box = _safe_bbox(xyxy, width, height, 0.0, min_size=7)
        depth_result = self._depth_from_crop(depth_raw, capture.depth_scale, crop_box)
        if depth_result is None and fallback_crop_box != crop_box:
            depth_result = self._depth_from_crop(depth_raw, capture.depth_scale, fallback_crop_box)

        if crop_box is None or depth_result is None:
            valid_count = 0 if depth_result is None else depth_result["valid_count"]
            return {
                "depth_m": None,
                "depth_pixel_count": valid_count,
                "depth_pixel": None,
                "camera_xyz": None,
                "world_xyz": None,
                "depth_crop": crop_box,
            }

        x1, y1, x2, y2 = depth_result["crop_box"]
        depth_m = depth_result["depth_m"]
        center_u = float((x1 + x2 - 1) / 2.0)
        center_v = float((y1 + y2 - 1) / 2.0)
        camera_xyz = capture.deproject_pixel((center_u, center_v), depth_m)
        world_xyz = _camera_to_world(camera_xyz, self.camera_pose)
        return {
            "depth_m": depth_m,
            "depth_pixel_count": depth_result["valid_count"],
            "depth_pixel": [center_u, center_v],
            "camera_xyz": camera_xyz,
            "world_xyz": world_xyz,
            "depth_crop": [x1, y1, x2, y2],
        }

    def _depth_from_crop(
        self,
        depth_raw: np.ndarray,
        depth_scale: float,
        crop_box: tuple[int, int, int, int] | None,
    ) -> dict[str, Any] | None:
        if crop_box is None:
            return None
        x1, y1, x2, y2 = crop_box
        crop = depth_raw[y1:y2, x1:x2].astype(np.float32) * depth_scale
        valid = crop[(crop >= self.depth_min) & (crop <= self.depth_max)]
        if valid.size < self.min_depth_pixels:
            return None
        return {
            "depth_m": float(np.median(valid)),
            "valid_count": int(valid.size),
            "crop_box": [x1, y1, x2, y2],
        }


def draw_detections(
    color_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    show_world: bool,
) -> np.ndarray:
    annotated = color_bgr.copy()
    for item in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in item["xyxy"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        if item["depth_crop"] is not None:
            cx1, cy1, cx2, cy2 = item["depth_crop"]
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 180, 0), 1)

        xyz = item["world_xyz"] if show_world else item["camera_xyz"]
        frame_name = "world" if show_world else "cam"
        if xyz is None:
            label = f"{item['class']} {item['confidence']:.2f} depth=None"
        else:
            label = (
                f"{item['class']} {item['confidence']:.2f} "
                f"{frame_name}=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})"
            )
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        if item["depth_pixel"] is not None:
            u, v = item["depth_pixel"]
            cv2.circle(annotated, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
    return annotated


def print_detections(frame_idx: int, detections: list[dict[str, Any]], *, show_world: bool) -> None:
    coord_name = "World XYZ" if show_world else "Camera XYZ"
    print(f"\n--- Frame {frame_idx} ---", flush=True)
    print(
        f"{'Class':<12} | {'Conf':<5} | {'Depth':<7} | {'Depth px':<16} | {coord_name:<28} | {'Valid depth px':<14}",
        flush=True,
    )
    for item in detections:
        xyz = item["world_xyz"] if show_world else item["camera_xyz"]
        depth = "None" if item["depth_m"] is None else f"{item['depth_m']:.3f}"
        print(
            f"{item['class']:<12} | {item['confidence']:<5.2f} | {depth:<7} | "
            f"{str(None if item['depth_pixel'] is None else np.round(item['depth_pixel'], 1).tolist()):<16} | "
            f"{_fmt_vec(xyz):<28} | {item['depth_pixel_count']:<14}",
            flush=True,
        )
    if not detections:
        print("No detections. Try lowering --conf or check the RGB image.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on RealSense RGB-D and print depth-based coordinates.")
    parser.add_argument("--model-path", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--device", default="", help="Ultralytics device, for example 0, cuda:0, or cpu. Default: auto.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--realsense-serial", default="", help="Optional RealSense serial number.")
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--bbox-shrink", type=float, default=0.25, help="Shrink bbox before sampling depth to avoid edges.")
    parser.add_argument("--min-depth-pixels", type=int, default=20)
    parser.add_argument("--camera-pose-json", default="", help="Optional camera_to_world transform JSON.")
    parser.add_argument("--camera-z", type=float, default=DEFAULT_CAMERA_Z)
    parser.add_argument("--show-world", action="store_true", help="Draw/print world_xyz instead of camera_xyz.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--output-dir", default="", help="Optional directory for annotated frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weight file not found: {model_path}")

    camera_pose = None
    if args.show_world or args.camera_pose_json:
        camera_pose = _load_camera_pose(args.camera_pose_json) if args.camera_pose_json else _default_camera_pose(args.camera_z)

    capture = RealSenseRgbdCapture(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.realsense_serial,
    )
    detector = RgbdYoloCoordinateDetector(
        str(model_path),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        bbox_shrink=args.bbox_shrink,
        min_depth_pixels=args.min_depth_pixels,
        camera_pose=camera_pose,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    intr = capture.color_intrinsics
    print(f"YOLO model: {model_path}", flush=True)
    print(f"RealSense RGB-D: {capture.width}x{capture.height}, depth_scale={capture.depth_scale}", flush=True)
    print(f"Color intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}", flush=True)
    print("Coordinate frame: camera_xyz uses RealSense optical frame, in meters.", flush=True)
    if camera_pose is not None:
        print(f"Camera pose:\n{np.round(camera_pose, 4)}", flush=True)

    processed = 0
    grabbed = 0
    try:
        while True:
            color_bgr, depth_raw = capture.read()
            grabbed += 1
            if args.frame_stride > 1 and (grabbed - 1) % args.frame_stride != 0:
                continue

            detections = detector.detect(color_bgr, depth_raw, capture)
            print_detections(processed, detections, show_world=args.show_world)

            annotated = draw_detections(color_bgr, detections, show_world=args.show_world)
            if output_dir is not None:
                out_path = output_dir / f"rgbd_yolo_{processed:04d}.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"Saved annotated frame: {out_path}", flush=True)

            if args.display:
                cv2.imshow("realsense_rgbd_yolo_coordinates", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break

            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            time.sleep(0.01)
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
