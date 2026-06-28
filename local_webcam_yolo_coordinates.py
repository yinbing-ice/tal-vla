#!/usr/bin/env python3
"""
Run the TAL YOLO detector on a local webcam and estimate plane-intersection coordinates.

This script is intentionally independent from Isaac Sim.  It is meant for quickly testing
the YOLO detection and coordinate back-projection logic with a real/local camera.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

LOCAL_CONFIG_DIR = Path(__file__).resolve().parent / ".runtime_config"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CONFIG_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CONFIG_DIR / "ultralytics"))
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(LOCAL_CONFIG_DIR / "ultralytics"))
LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
(LOCAL_CONFIG_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(LOCAL_CONFIG_DIR / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_YOLO_MODEL = r"D:\code\weight\best.pt"
DEFAULT_ROOM_BOUNDS = (-5.0, 5.0, -5.0, 5.0)
DEFAULT_CAMERA_Z = 1.50
DEFAULT_TABLE_Z = 0.53
DEFAULT_GROUND_Z = 0.00
DEFAULT_FPS = 30
YOLO_CLASS_NAMES = ["robot", "cube", "bottle", "table", "stool", "smallpallet", "bigpallet"]
DEFAULT_TABLETOP_CLASSES = {"cube", "bottle", "smallpallet", "bigpallet"}


def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _inside_xy_bounds(point: np.ndarray, bounds: dict[str, float]) -> bool:
    return bounds["x_min"] <= point[0] <= bounds["x_max"] and bounds["y_min"] <= point[1] <= bounds["y_max"]


def _inside_image_bbox(u: float, v: float, bbox: list[float], margin_px: float = 0.0) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 - margin_px <= u <= x2 + margin_px and y1 - margin_px <= v <= y2 + margin_px


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


def _default_overhead_pose(camera_z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.0, 0.0, float(camera_z)]
    return pose


def _model_class_name(model_names: dict[int, str] | list[str], cls_id: int) -> str:
    if isinstance(model_names, dict) and cls_id in model_names:
        return str(model_names[cls_id]).lower()
    if isinstance(model_names, list) and 0 <= cls_id < len(model_names):
        return str(model_names[cls_id]).lower()
    if 0 <= cls_id < len(YOLO_CLASS_NAMES):
        return YOLO_CLASS_NAMES[cls_id]
    return str(cls_id)


class OpenCVCapture:
    def __init__(self, source: int | str, *, width: int, height: int, fps: int) -> None:
        api_preference = cv2.CAP_DSHOW if os.name == "nt" and isinstance(source, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, api_preference)
        self.backend_name = "opencv/dshow" if api_preference == cv2.CAP_DSHOW else "opencv"
        if not self.cap.isOpened() and api_preference != cv2.CAP_ANY:
            self.cap.release()
            self.cap = cv2.VideoCapture(source)
            self.backend_name = "opencv"
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera/video source: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
        self.warning = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def intrinsics(self) -> dict[str, float]:
        return {}


class RealSenseCapture:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        stream: str,
        serial: str,
        timeout_ms: int = 5000,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is not installed. Install it with: pip install pyrealsense2") from exc

        self.rs = rs
        self.timeout_ms = int(timeout_ms)
        self.pipeline = None
        self.profile = None
        self.stream_kind = ""
        self.width = int(width)
        self.height = int(height)
        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self.warning = None

        stream_order = [stream] if stream != "auto" else ["color", "depth", "infrared"]
        errors = []
        for stream_kind in stream_order:
            for use_requested_size in (True, False):
                try:
                    self.pipeline, self.profile = self._start_stream(
                        stream_kind,
                        serial=serial,
                        width=width,
                        height=height,
                        fps=fps,
                        use_requested_size=use_requested_size,
                    )
                    self.stream_kind = stream_kind
                    self._read_intrinsics()
                    self.backend_name = f"realsense/{self.stream_kind}"
                    if self.stream_kind != "color":
                        self.warning = (
                            f"RealSense is using {self.stream_kind} frames. "
                            "RGB-trained YOLO weights usually work best with a color stream."
                        )
                    return
                except Exception as exc:
                    errors.append(f"{stream_kind} requested_size={use_requested_size}: {exc}")
                    if self.pipeline is not None:
                        try:
                            self.pipeline.stop()
                        except Exception:
                            pass
                    self.pipeline = None
                    self.profile = None

        joined = "\n  - ".join(errors)
        raise RuntimeError(f"Failed to start RealSense stream.\n  - {joined}")

    def _start_stream(
        self,
        stream_kind: str,
        *,
        serial: str,
        width: int,
        height: int,
        fps: int,
        use_requested_size: bool,
    ):
        rs = self.rs
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)

        if stream_kind == "color":
            if use_requested_size:
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            else:
                config.enable_stream(rs.stream.color)
        elif stream_kind == "depth":
            if use_requested_size:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            else:
                config.enable_stream(rs.stream.depth)
        elif stream_kind == "infrared":
            if use_requested_size:
                config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            else:
                config.enable_stream(rs.stream.infrared, 1)
        else:
            raise ValueError(f"Unsupported RealSense stream: {stream_kind}")

        profile = pipeline.start(config)
        return pipeline, profile

    def _read_intrinsics(self) -> None:
        if self.profile is None:
            return
        target = {
            "color": self.rs.stream.color,
            "depth": self.rs.stream.depth,
            "infrared": self.rs.stream.infrared,
        }[self.stream_kind]
        for stream_profile in self.profile.get_streams():
            if stream_profile.stream_type() != target:
                continue
            video_profile = stream_profile.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            self.width = int(intr.width)
            self.height = int(intr.height)
            self.fx = float(intr.fx)
            self.fy = float(intr.fy)
            self.cx = float(intr.ppx)
            self.cy = float(intr.ppy)
            return

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.pipeline is None:
            return False, None
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if self.stream_kind == "color":
            frame = frames.get_color_frame()
            if not frame:
                return False, None
            image = np.asanyarray(frame.get_data())
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return True, image

        if self.stream_kind == "depth":
            frame = frames.get_depth_frame()
            if not frame:
                return False, None
            depth = np.asanyarray(frame.get_data())
            depth_8u = cv2.convertScaleAbs(depth, alpha=0.03)
            return True, cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        frame = frames.get_infrared_frame(1)
        if not frame:
            return False, None
        infrared = np.asanyarray(frame.get_data())
        return True, cv2.cvtColor(infrared, cv2.COLOR_GRAY2BGR)

    def release(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def intrinsics(self) -> dict[str, float]:
        data = {}
        if self.fx is not None:
            data["fx"] = self.fx
        if self.fy is not None:
            data["fy"] = self.fy
        if self.cx is not None:
            data["cx"] = self.cx
        if self.cy is not None:
            data["cy"] = self.cy
        return data


def open_capture(args: argparse.Namespace) -> OpenCVCapture | RealSenseCapture:
    source = _parse_source(args.source)
    source_name = str(args.source).strip().lower()
    real_sense_source = source_name in {"realsense", "rs", "d405", "d435", "435"}
    should_try_realsense = args.camera_backend == "realsense" or (
        args.camera_backend == "auto" and (real_sense_source or source_name == "0")
    )

    if should_try_realsense:
        try:
            return RealSenseCapture(
                width=args.width,
                height=args.height,
                fps=args.fps,
                stream=args.realsense_stream,
                serial=args.realsense_serial,
            )
        except Exception:
            if args.camera_backend == "realsense" or real_sense_source:
                raise

    return OpenCVCapture(source, width=args.width, height=args.height, fps=args.fps)


class LocalWebcamYoloCoordinateTester:
    def __init__(
        self,
        model_path: str,
        *,
        width: int,
        height: int,
        conf: float,
        device: str,
        fx: float | None,
        fy: float | None,
        cx: float | None,
        cy: float | None,
        hfov_deg: float,
        camera_pose: np.ndarray,
        table_z: float,
        ground_z: float,
        room_bounds: tuple[float, float, float, float],
    ) -> None:
        self.model = YOLO(model_path)
        self.width = int(width)
        self.height = int(height)
        self.conf = float(conf)
        self.device = device.strip()
        self.fx = float(fx) if fx is not None else self._fx_from_hfov(hfov_deg)
        self.fy = float(fy) if fy is not None else self.fx
        self.cx = float(cx) if cx is not None else self.width / 2.0
        self.cy = float(cy) if cy is not None else self.height / 2.0
        self.camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
        self.table_z = float(table_z)
        self.ground_z = float(ground_z)
        self.room_bounds = tuple(float(v) for v in room_bounds)

    def _fx_from_hfov(self, hfov_deg: float) -> float:
        hfov_rad = math.radians(float(hfov_deg))
        return self.width / (2.0 * math.tan(hfov_rad / 2.0))

    def detect_image(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        predict_kwargs = {"source": image_bgr, "conf": self.conf, "verbose": False}
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
            detections.append(
                {
                    "class": class_name,
                    "confidence": float(box.conf[0].item()),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "xywh": [float(u), float(v), float(w), float(h)],
                }
            )
        return detections

    def estimate_positions(self, detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
        table_info = self._estimate_table_info(detections)
        table_bounds = None if table_info is None else table_info["world_bounds"]
        table_bbox = None if table_info is None else table_info["image_bbox"]
        estimates = []

        for det in detections:
            support_u, support_v = self._support_pixel(det)
            table_xyz = self.pixel_to_world_on_plane(support_u, support_v, self.table_z)
            ground_xyz = self.pixel_to_world_on_plane(support_u, support_v, self.ground_z)
            selected_plane, selected_xyz = self._select_support_plane(
                det["class"],
                support_u,
                support_v,
                table_xyz,
                ground_xyz,
                table_bounds,
                table_bbox,
            )
            estimates.append(
                {
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "xyxy": det["xyxy"],
                    "support_pixel": [support_u, support_v],
                    "selected_plane": selected_plane,
                    "selected_xyz": selected_xyz,
                    "table_xyz": table_xyz,
                    "ground_xyz": ground_xyz,
                }
            )
        return estimates, table_bounds

    def pixel_to_world_on_plane(self, u: float, v: float, target_z: float) -> np.ndarray | None:
        x_c = (u - self.cx) / self.fx
        y_c = -(v - self.cy) / self.fy
        ray_c = np.array([x_c, y_c, -1.0], dtype=np.float64)
        ray_c /= np.linalg.norm(ray_c)

        ray_w = self.camera_pose[:3, :3] @ ray_c
        cam_origin = self.camera_pose[:3, 3]
        if abs(ray_w[2]) < 1e-8:
            return None
        t = (target_z - cam_origin[2]) / ray_w[2]
        if t <= 0:
            return None
        return cam_origin + t * ray_w

    @staticmethod
    def _support_pixel(det: dict[str, Any]) -> tuple[float, float]:
        x1, _, x2, y2 = det["xyxy"]
        u, v, _, _ = det["xywh"]
        if det["class"] in {"table", "robot"}:
            return float(u), float(v)
        return float((x1 + x2) / 2.0), float(y2)

    def _estimate_table_info(self, detections: list[dict[str, Any]], margin: float = 0.25) -> dict[str, Any] | None:
        table_candidates = [det for det in detections if "table" in det["class"]]
        if not table_candidates:
            return None
        table = max(table_candidates, key=lambda det: det["confidence"])
        x1, y1, x2, y2 = table["xyxy"]
        points = [
            self.pixel_to_world_on_plane(u, v, self.table_z)
            for u, v in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        ]
        points = [point for point in points if point is not None and self._inside_room(point)]
        if not points:
            return None
        points_np = np.asarray(points, dtype=np.float64)
        return {
            "world_bounds": {
                "x_min": float(points_np[:, 0].min() - margin),
                "x_max": float(points_np[:, 0].max() + margin),
                "y_min": float(points_np[:, 1].min() - margin),
                "y_max": float(points_np[:, 1].max() + margin),
            },
            "image_bbox": [float(x1), float(y1), float(x2), float(y2)],
        }

    def _select_support_plane(
        self,
        class_name: str,
        support_u: float,
        support_v: float,
        table_xyz: np.ndarray | None,
        ground_xyz: np.ndarray | None,
        table_bounds: dict[str, float] | None,
        table_bbox: list[float] | None,
    ) -> tuple[str, np.ndarray | None]:
        if class_name in {"robot", "table", "stool"}:
            return "ground", ground_xyz
        if table_bbox is not None and _inside_image_bbox(support_u, support_v, table_bbox, margin_px=20):
            return "table", table_xyz
        if class_name in DEFAULT_TABLETOP_CLASSES and table_xyz is not None:
            return "table", table_xyz
        if table_xyz is not None and self._inside_room(table_xyz):
            if table_bounds is None or _inside_xy_bounds(table_xyz, table_bounds):
                return "table", table_xyz
        return "ground", ground_xyz

    def _inside_room(self, point: np.ndarray) -> bool:
        x_min, x_max, y_min, y_max = self.room_bounds
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def draw_estimates(image_bgr: np.ndarray, estimates: list[dict[str, Any]]) -> np.ndarray:
    annotated = image_bgr.copy()
    for item in estimates:
        x1, y1, x2, y2 = [int(round(v)) for v in item["xyxy"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{item['class']} {item['confidence']:.2f} {item['selected_plane']}"
        if item["selected_xyz"] is not None:
            xyz = np.asarray(item["selected_xyz"], dtype=float)
            label += f" xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})"
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
        support_u, support_v = item["support_pixel"]
        cv2.circle(annotated, (int(round(support_u)), int(round(support_v))), 4, (0, 0, 255), -1)
    return annotated


def print_estimates(frame_idx: int, estimates: list[dict[str, Any]], table_bounds: dict[str, float] | None) -> None:
    print(f"\n--- Frame {frame_idx} ---", flush=True)
    print(f"Table XY bounds from detection: {table_bounds}", flush=True)
    print(f"{'Class':<12} | {'Conf':<5} | {'Plane':<6} | {'Support px':<18} | {'Selected XYZ':<28}", flush=True)
    for item in estimates:
        print(
            f"{item['class']:<12} | {item['confidence']:<5.2f} | {item['selected_plane']:<6} | "
            f"{np.round(item['support_pixel'], 1).tolist()!s:<18} | {_fmt_vec(item['selected_xyz']):<28}",
            flush=True,
        )
    if not estimates:
        print("No detections. Try lowering --conf or check the camera image.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAL YOLO on a local webcam and print approximate plane-intersection coordinates."
    )
    parser.add_argument(
        "--source",
        default="realsense",
        help="Camera source. Use 'realsense' for Intel RealSense, or an OpenCV index/file/URL. Default: realsense",
    )
    parser.add_argument("--model-path", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="", help="Ultralytics device, for example 0, cuda:0, or cpu. Default: auto.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--camera-backend", choices=["auto", "realsense", "opencv"], default="auto")
    parser.add_argument("--realsense-stream", choices=["auto", "color", "depth", "infrared"], default="auto")
    parser.add_argument("--realsense-serial", default="", help="Optional RealSense serial number.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window.")
    parser.add_argument("--output-dir", default="", help="Optional directory for annotated frames.")

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--hfov-deg", type=float, default=70.0)
    parser.add_argument("--camera-pose-json", default="", help="JSON with camera_to_world or Isaac cameraViewTransform.")
    parser.add_argument("--camera-z", type=float, default=DEFAULT_CAMERA_Z, help="Fallback overhead camera z when no pose JSON is given.")
    parser.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z)
    parser.add_argument("--ground-z", type=float, default=DEFAULT_GROUND_Z)
    parser.add_argument("--room-bounds", type=float, nargs=4, default=DEFAULT_ROOM_BOUNDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weight file not found: {model_path}")

    # 2026-06-25 修改：本脚本用于脱离 Isaac，用服务器本地摄像头快速验证 YOLO 检测和坐标反投影。
    # 若没有真实相机外参，默认只能得到相机坐标系/自定义平面下的近似坐标，真实世界坐标需要标定。
    camera_pose = _load_camera_pose(args.camera_pose_json) if args.camera_pose_json else _default_overhead_pose(args.camera_z)

    cap = open_capture(args)
    intrinsics = cap.intrinsics()
    fx = args.fx if args.fx is not None else intrinsics.get("fx")
    fy = args.fy if args.fy is not None else intrinsics.get("fy")
    cx = args.cx if args.cx is not None else intrinsics.get("cx")
    cy = args.cy if args.cy is not None else intrinsics.get("cy")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    tester = LocalWebcamYoloCoordinateTester(
        str(model_path),
        width=cap.width,
        height=cap.height,
        conf=args.conf,
        device=args.device,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        hfov_deg=args.hfov_deg,
        camera_pose=camera_pose,
        table_z=args.table_z,
        ground_z=args.ground_z,
        room_bounds=tuple(args.room_bounds),
    )

    print(f"YOLO model: {model_path}", flush=True)
    print(f"Source: {args.source}", flush=True)
    print(f"Camera backend: {cap.backend_name}", flush=True)
    print(f"Frame size: {cap.width}x{cap.height}", flush=True)
    if cap.warning:
        print(f"Warning: {cap.warning}", flush=True)
    print(
        "Intrinsics: "
        f"fx={tester.fx:.2f}, fy={tester.fy:.2f}, cx={tester.cx:.1f}, cy={tester.cy:.1f}",
        flush=True,
    )
    print(f"Camera pose:\n{np.round(camera_pose, 4)}", flush=True)

    processed = 0
    grabbed = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("Camera/video source returned no frame")
            grabbed += 1
            if args.frame_stride > 1 and (grabbed - 1) % args.frame_stride != 0:
                continue

            if frame_bgr.shape[1] != tester.width or frame_bgr.shape[0] != tester.height:
                frame_bgr = cv2.resize(frame_bgr, (tester.width, tester.height))

            detections = tester.detect_image(frame_bgr)
            estimates, table_bounds = tester.estimate_positions(detections)
            print_estimates(processed, estimates, table_bounds)

            annotated = draw_estimates(frame_bgr, estimates)
            if output_dir is not None:
                out_path = output_dir / f"webcam_yolo_{processed:04d}.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"Saved annotated frame: {out_path}", flush=True)

            if args.display:
                cv2.imshow("local_webcam_yolo_coordinates", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break

            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            time.sleep(0.01)
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
