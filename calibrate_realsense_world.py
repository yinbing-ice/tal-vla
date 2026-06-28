#!/usr/bin/env python3
"""
Calibrate a RealSense RGB-D camera to a user-defined world coordinate frame.

Click known points in the RGB image, type their world XYZ coordinates in meters,
and this script solves the rigid camera_to_world transform:

    world_xyz = R @ camera_xyz + t

The saved JSON can be passed to realsense_rgbd_yolo_coordinates.py with
--camera-pose-json and --show-world.
桌子长宽高48*70*52
0,0,0.52
-0.23,-0.33,0.52
-0.23, 0.0, 0.52
0.0, 0.35, 0.52s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_FPS = 30
WINDOW_NAME = "calibrate_realsense_world"


def _fmt_vec(value: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return str(np.round(np.asarray(value, dtype=float), 4).tolist())


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


def estimate_depth_around_pixel(
    depth_raw: np.ndarray,
    *,
    u: int,
    v: int,
    depth_scale: float,
    radius: int,
    depth_min: float,
    depth_max: float,
    min_depth_pixels: int,
) -> tuple[float | None, int]:
    height, width = depth_raw.shape[:2]
    x1 = max(0, int(u) - radius)
    y1 = max(0, int(v) - radius)
    x2 = min(width, int(u) + radius + 1)
    y2 = min(height, int(v) + radius + 1)
    crop = depth_raw[y1:y2, x1:x2].astype(np.float32) * float(depth_scale)
    valid = crop[(crop >= depth_min) & (crop <= depth_max)]
    if valid.size < min_depth_pixels:
        return None, int(valid.size)
    return float(np.median(valid)), int(valid.size)


def solve_rigid_transform(camera_points: np.ndarray, world_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if camera_points.shape != world_points.shape or camera_points.ndim != 2 or camera_points.shape[1] != 3:
        raise ValueError("camera_points and world_points must both have shape Nx3")
    if camera_points.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required")

    camera_centroid = camera_points.mean(axis=0)
    world_centroid = world_points.mean(axis=0)
    camera_centered = camera_points - camera_centroid
    world_centered = world_points - world_centroid

    h_mat = camera_centered.T @ world_centered
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    translation = world_centroid - rotation @ camera_centroid

    predicted = (rotation @ camera_points.T).T + translation
    errors = np.linalg.norm(predicted - world_points, axis=1)
    return rotation, translation, float(errors.mean()), errors


def parse_world_xyz(raw: str) -> np.ndarray | None:
    text = raw.strip().replace(",", " ")
    if not text:
        return None
    parts = text.split()
    if len(parts) != 3:
        raise ValueError("Please enter exactly three numbers: x y z")
    return np.asarray([float(v) for v in parts], dtype=np.float64)


def draw_overlay(
    frame_bgr: np.ndarray,
    samples: list[dict[str, Any]],
    pending_click: tuple[int, int] | None,
) -> np.ndarray:
    image = frame_bgr.copy()
    for idx, item in enumerate(samples, start=1):
        u, v = item["pixel"]
        cv2.circle(image, (int(u), int(v)), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(idx),
            (int(u) + 7, int(v) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if pending_click is not None:
        cv2.circle(image, pending_click, 6, (255, 180, 0), 2)

    lines = [
        "Click known point; type world XYZ in terminal.",
        "Keys: s=solve/save, u=undo, q=quit",
        f"Samples: {len(samples)}",
    ]
    for idx, line in enumerate(lines):
        y = 24 + idx * 24
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_camera_pose(
    path: Path,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    mean_error: float,
    errors: np.ndarray,
    samples: list[dict[str, Any]],
) -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    payload = {
        "camera_to_world": pose.tolist(),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "mean_error_m": mean_error,
        "per_point_error_m": errors.tolist(),
        "samples": [
            {
                "pixel": item["pixel"],
                "depth_m": item["depth_m"],
                "camera_xyz": item["camera_xyz"].tolist(),
                "world_xyz": item["world_xyz"].tolist(),
            }
            for item in samples
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click known RGB-D points and solve camera_to_world transform.")
    parser.add_argument("--output", default="camera_pose.json", help="Output JSON path.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--realsense-serial", default="", help="Optional RealSense serial number.")
    parser.add_argument("--depth-radius", type=int, default=4, help="Median depth window radius around clicked pixel.")
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--min-depth-pixels", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    capture = RealSenseRgbdCapture(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.realsense_serial,
    )
    samples: list[dict[str, Any]] = []
    pending_click: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        del flags, userdata
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = (int(x), int(y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Click a known point in the image, then enter its world XYZ in meters.", flush=True)
    print("Use at least 3 non-collinear points; 6+ points is better.", flush=True)
    print("Example world XYZ: 0.50 0.20 0.53", flush=True)

    last_color = None
    last_depth = None
    try:
        while True:
            color_bgr, depth_raw = capture.read()
            last_color = color_bgr
            last_depth = depth_raw

            if pending_click is not None:
                u, v = pending_click
                depth_m, valid_count = estimate_depth_around_pixel(
                    depth_raw,
                    u=u,
                    v=v,
                    depth_scale=capture.depth_scale,
                    radius=args.depth_radius,
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                    min_depth_pixels=args.min_depth_pixels,
                )
                if depth_m is None:
                    print(
                        f"No reliable depth at pixel ({u}, {v}); valid depth pixels={valid_count}. Try another point.",
                        flush=True,
                    )
                    pending_click = None
                else:
                    camera_xyz = capture.deproject_pixel((u, v), depth_m)
                    print(f"\nClicked pixel=({u}, {v}), depth={depth_m:.4f} m, camera_xyz={_fmt_vec(camera_xyz)}", flush=True)
                    while True:
                        raw = input("World XYZ in meters, or blank to cancel this point: ")
                        try:
                            world_xyz = parse_world_xyz(raw)
                            break
                        except ValueError as exc:
                            print(f"Invalid input: {exc}", flush=True)
                    if world_xyz is not None:
                        samples.append(
                            {
                                "pixel": [int(u), int(v)],
                                "depth_m": float(depth_m),
                                "camera_xyz": camera_xyz,
                                "world_xyz": world_xyz,
                            }
                        )
                        print(f"Added sample #{len(samples)}: world_xyz={_fmt_vec(world_xyz)}", flush=True)
                    pending_click = None

            display = draw_overlay(color_bgr, samples, pending_click)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("u"):
                if samples:
                    removed = samples.pop()
                    print(f"Removed sample at pixel={removed['pixel']}", flush=True)
            if key == ord("s"):
                if len(samples) < 3:
                    print("Need at least 3 samples before solving.", flush=True)
                    continue
                camera_points = np.asarray([item["camera_xyz"] for item in samples], dtype=np.float64)
                world_points = np.asarray([item["world_xyz"] for item in samples], dtype=np.float64)
                rotation, translation, mean_error, errors = solve_rigid_transform(camera_points, world_points)
                save_camera_pose(
                    output_path,
                    rotation=rotation,
                    translation=translation,
                    mean_error=mean_error,
                    errors=errors,
                    samples=samples,
                )
                print(f"\nSaved camera pose: {output_path}", flush=True)
                print(f"Mean calibration error: {mean_error:.4f} m", flush=True)
                print(f"Per-point errors: {_fmt_vec(errors)}", flush=True)
                print(f"camera_to_world rotation:\n{np.round(rotation, 6)}", flush=True)
                print(f"camera_to_world translation: {_fmt_vec(translation)}", flush=True)
                print("You can now use: --camera-pose-json camera_pose.json --show-world", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if last_color is None or last_depth is None:
        print("No frames were captured.", flush=True)


if __name__ == "__main__":
    main()
