import os
import time
from copy import deepcopy

import numpy as np


# 2026-05-10 修改：将训练数据中的物体坐标从 Isaac 上帝视角改为 YOLO + 相机反投影观测。
# 机器人本体和已知静态桌子仍使用仿真真值，分别对应真实机器人自定位和已知场景标定。
DEFAULT_YOLO_MODEL_PATH = "/root/gpufree-data/PRJ/Yolo2/runs/detect/train/weights/best.pt"
# 2026-06-08 修改：high 第一人称相机已挂到小车下面，YOLO 反投影默认也读取新的车载相机位姿。
# 若场景使用其他感知相机，可继续通过 TAL_YOLO_CAMERA_PATH 覆盖。
DEFAULT_CAMERA_PATH = "/World/Mobie_grasper2/high"
DEFAULT_TABLE_Z = -1.40
DEFAULT_GROUND_Z = -1.93
DEFAULT_ROOM_BOUNDS = (-3.76, 2.09, -3.79, -0.153)
DEFAULT_OBJECT_ORIENTATION = [0.0, 0.0, 0.0, 1.0]
DEFAULT_CAPTURE_TIMEOUT_S = 8.0

YOLO_CLASS_NAMES = ["robot", "cube", "bottle", "table", "stool", "smallpallet", "bigpallet"]


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _yolo_debug(message):
    # 2026-05-11 修改：默认关闭 YOLO 感知细节日志，避免 exploration 运行时终端刷屏。
    # 需要查看抓图/检测耗时时设置 TAL_YOLO_DEBUG=1。
    if _env_flag("TAL_YOLO_DEBUG", False):
        print(f"[YOLOPerception] {message}", flush=True)


def perception_enabled():
    return os.environ.get("TAL_USE_YOLO_PERCEPTION", "1").lower() not in {"0", "false", "no", "off"}


def apply_yolo_observation_to_datapoint(config, datapoint, true_metrics=None, constraints=None, image_rgb=None):
    """Replace datapoint metrics with YOLO-observed positions where available."""
    if not perception_enabled():
        return datapoint

    observed_metrics = get_observed_metrics(
        config,
        true_metrics=true_metrics,
        constraints=constraints,
        image_rgb=image_rgb,
    )
    if not observed_metrics:
        return datapoint

    observed_datapoint = datapoint.deepcopy()
    for idx, metric_snapshot in enumerate(observed_datapoint.metrics):
        if not isinstance(metric_snapshot, dict):
            continue
        updated_snapshot = deepcopy(metric_snapshot)
        for obj_name, observed_metric in observed_metrics.items():
            if obj_name in updated_snapshot:
                updated_snapshot[obj_name] = deepcopy(observed_metric)
        observed_datapoint.metrics[idx] = updated_snapshot

        if "husky" in updated_snapshot and idx < len(observed_datapoint.position):
            robot_pos = updated_snapshot["husky"][0]
            observed_datapoint.position[idx] = [robot_pos[0], robot_pos[1], robot_pos[2], 0.0]

    return observed_datapoint


def get_observed_metrics(config, true_metrics=None, constraints=None, image_rgb=None):
    """Return TAL metrics using robot self-localization + YOLO object observations."""
    if true_metrics is None:
        true_metrics = {}

    # 2026-05-10 修改：默认保留真值作为漏检兜底；设置 TAL_YOLO_PERCEPTION_FALLBACK_TRUE=0 可只使用检测结果。
    fallback_true = os.environ.get("TAL_YOLO_PERCEPTION_FALLBACK_TRUE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    observed_metrics = deepcopy(true_metrics) if fallback_true else {}

    for obj_name in ["husky", "table"]:
        if obj_name in true_metrics:
            observed_metrics[obj_name] = deepcopy(true_metrics[obj_name])

    try:
        detector = _get_detector(config)
        detections = detector.detect(image_rgb=image_rgb)
    except Exception as exc:
        # 2026-05-11 修改：感知失败会影响训练数据质量，因此 warning 默认保留；细节日志仍由 TAL_YOLO_DEBUG 控制。
        print(f"[YOLOPerception] warning: perception failed, keep fallback metrics. reason={exc}", flush=True)
        return observed_metrics

    held_obj = _held_object(constraints)
    for obj_name, xyz in detections.items():
        if obj_name in {"husky", "table"}:
            continue
        if held_obj == obj_name and obj_name in true_metrics:
            observed_metrics[obj_name] = deepcopy(true_metrics[obj_name])
            continue

        # 2026-05-12 修改：YOLO 当前只估计物体 3D 坐标，不估计姿态。
        # 因此所有 YOLO 观测到的可移动/普通物体统一使用单位四元数占位，避免训练 graph 混入 Isaac 姿态真值。
        # husky 和 table 在上面已保留真值/标定，不会走到这里。
        orientation = DEFAULT_OBJECT_ORIENTATION
        observed_metrics[obj_name] = [list(np.asarray(xyz, dtype=float)), list(orientation)]

    return observed_metrics


_DETECTOR = None


def _get_detector(config):
    global _DETECTOR
    model_path = os.environ.get("TAL_YOLO_MODEL_PATH", DEFAULT_YOLO_MODEL_PATH)
    camera_path = os.environ.get("TAL_YOLO_CAMERA_PATH", DEFAULT_CAMERA_PATH)
    if _DETECTOR is None or _DETECTOR.model_path != model_path or _DETECTOR.camera_path != camera_path:
        _DETECTOR = IsaacYoloCoordinateDetector(
            config=config,
            model_path=model_path,
            camera_path=camera_path,
            table_z=float(os.environ.get("TAL_YOLO_TABLE_Z", DEFAULT_TABLE_Z)),
            ground_z=float(os.environ.get("TAL_YOLO_GROUND_Z", DEFAULT_GROUND_Z)),
            room_bounds=_read_room_bounds(),
        )
    return _DETECTOR


def _read_room_bounds():
    raw = os.environ.get("TAL_YOLO_ROOM_BOUNDS")
    if not raw:
        return DEFAULT_ROOM_BOUNDS
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError("TAL_YOLO_ROOM_BOUNDS must contain 4 values: x_min x_max y_min y_max")
    return tuple(values)


def _held_object(constraints):
    if not isinstance(constraints, dict):
        return None
    held = constraints.get("husky", [])
    if isinstance(held, (list, tuple)) and held:
        return held[0]
    return None


class IsaacYoloCoordinateDetector:
    def __init__(self, config, model_path, camera_path, table_z, ground_z, room_bounds):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("ultralytics is required for TAL_USE_YOLO_PERCEPTION=1") from exc

        self.config = config
        self.model_path = model_path
        self.camera_path = camera_path
        self.model = YOLO(model_path)
        self.table_z = table_z
        self.ground_z = ground_z
        self.room_bounds = tuple(room_bounds)
        self.width = int(os.environ.get("TAL_YOLO_IMAGE_WIDTH", "1024"))
        self.height = int(os.environ.get("TAL_YOLO_IMAGE_HEIGHT", "1024"))
        self.conf = float(os.environ.get("TAL_YOLO_CONF", "0.35"))
        self.capture_timeout_s = float(os.environ.get("TAL_YOLO_CAPTURE_TIMEOUT_S", DEFAULT_CAPTURE_TIMEOUT_S))
        self.capture_mode = os.environ.get("TAL_YOLO_CAPTURE_MODE", "auto").strip().lower()
        self._live_camera = None
        self.fx = self.fy = None
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def detect(self, image_rgb=None):
        start_time = time.monotonic()
        _yolo_debug("start detection")
        stage, camera_prim, UsdGeom = self._get_stage_camera()
        self._configure_intrinsics(camera_prim)
        if image_rgb is None:
            _yolo_debug("camera ready, start RGB capture")
            image_rgb = self._capture_rgb()
        else:
            _yolo_debug("reuse online control cam_high frame for YOLO detection")
        if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
            raise RuntimeError("Isaac camera returned empty RGB frame")
        _yolo_debug(
            "RGB captured: "
            f"shape={getattr(image_rgb, 'shape', None)} dtype={getattr(image_rgb, 'dtype', None)}"
        )

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
            image_rgb = image_rgb[:, :, :3]

        camera_pose = _gf_matrix_to_column_pose(UsdGeom.XformCache().GetLocalToWorldTransform(camera_prim))
        _yolo_debug("start YOLO inference")
        raw_detections = self._detect_image(image_rgb)
        _yolo_debug(f"YOLO detections={len(raw_detections)}")
        table_info = self._estimate_table_info(raw_detections, camera_pose)
        detections = self._estimate_positions(raw_detections, camera_pose, table_info)
        _yolo_debug(
            f"finished in {time.monotonic() - start_time:.2f}s, "
            f"objects={sorted(detections.keys())}"
        )
        return detections

    def _get_stage_camera(self):
        try:
            from isaacsim.core.utils.stage import get_current_stage
        except ModuleNotFoundError:
            from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdGeom

        stage = get_current_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"Camera prim not found: {self.camera_path}")
        return stage, camera_prim, UsdGeom

    def _configure_intrinsics(self, camera_prim):
        focal = _get_usd_attr(camera_prim, "focalLength", 24.0)
        horizontal_aperture = _get_usd_attr(camera_prim, "horizontalAperture", 20.955)
        vertical_aperture = _get_usd_attr(camera_prim, "verticalAperture", horizontal_aperture)
        self.fx = float(focal) / float(horizontal_aperture) * self.width
        self.fy = float(focal) / float(vertical_aperture) * self.height
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def _capture_rgb(self):
        # 2026-05-18 修改：在线控制阶段优先读取当前 live camera 帧，避免 Replicator orchestrator.step()
        # 额外推进仿真时间线，导致机械臂姿态在 TAL/YOLO 重规划前后发生突变。
        if self.capture_mode in {"auto", "live_camera", "live"}:
            image = self._capture_rgb_with_live_camera()
            if image is not None and getattr(image, "size", 0) != 0:
                return np.asarray(image)
        if self.capture_mode in {"auto", "replicator", "rep"}:
            image = self._capture_rgb_with_replicator()
            if image is not None and getattr(image, "size", 0) != 0:
                return np.asarray(image)
        return None

    def _capture_rgb_with_live_camera(self):
        try:
            import omni.kit.app
            try:
                from isaacsim.sensors.camera import Camera
            except ModuleNotFoundError:
                from omni.isaac.sensor import Camera
        except Exception:
            return None

        if self._live_camera is None:
            try:
                self._live_camera = Camera(prim_path=self.camera_path, resolution=(self.width, self.height))
                self._live_camera.initialize()
            except Exception as exc:
                _yolo_debug(f"live camera init failed: {exc}")
                self._live_camera = None
                return None

        app = omni.kit.app.get_app()
        for _ in range(2):
            app.update()

        try:
            image = self._live_camera.get_rgba()
        except Exception as exc:
            _yolo_debug(f"live camera get_rgba failed: {exc}")
            return None

        if image is None or getattr(image, "size", 0) == 0:
            _yolo_debug("live camera returned empty frame")
            return None

        image = np.asarray(image)
        _yolo_debug(
            "Live camera frame: "
            f"shape={getattr(image, 'shape', None)} dtype={getattr(image, 'dtype', None)} size={getattr(image, 'size', None)}"
        )
        return image

    def _capture_rgb_with_replicator(self):
        import omni.kit.app
        import omni.replicator.core as rep

        _yolo_debug(
            f"Replicator capture start: camera={self.camera_path} "
            f"resolution=({self.width}, {self.height}) timeout={self.capture_timeout_s}s"
        )
        app = omni.kit.app.get_app()
        render_product = rep.create.render_product(self.camera_path, resolution=(self.width, self.height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach(render_product)
        try:
            rep.orchestrator.set_capture_on_play(False)
            image = None
            warmup_steps = int(os.environ.get("TAL_YOLO_WARMUP_STEPS", "4"))
            capture_start = time.monotonic()
            attempt = 0
            while time.monotonic() - capture_start < self.capture_timeout_s:
                attempt += 1
                try:
                    rep.orchestrator.step(rt_subframes=max(warmup_steps, 1))
                except TypeError:
                    rep.orchestrator.step()
                for _ in range(2):
                    app.update()
                image = rgb_annotator.get_data()
                _yolo_debug(
                    "Replicator attempt "
                    f"{attempt}: shape={getattr(image, 'shape', None)} "
                    f"dtype={getattr(image, 'dtype', None)} size={getattr(image, 'size', None)}"
                )
                if image is not None and getattr(image, "size", 0) != 0:
                    return np.asarray(image)
            _yolo_debug("Replicator capture timeout")
            return image
        finally:
            try:
                rgb_annotator.detach()
            except Exception:
                pass
            try:
                render_product.destroy()
            except Exception:
                pass

    def _detect_image(self, image_rgb):
        results = self.model.predict(source=image_rgb, conf=self.conf, verbose=False)[0]
        model_names = getattr(self.model, "names", None) or {}
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(model_names.get(cls_id, YOLO_CLASS_NAMES[cls_id])).lower()
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

    def _estimate_positions(self, detections, camera_pose, table_info):
        table_bounds = None if table_info is None else table_info["world_bounds"]
        table_bbox = None if table_info is None else table_info["image_bbox"]
        best_by_object = {}

        for det in detections:
            class_name = det["class"]
            tal_name = self._resolve_tal_name(class_name, "ground")
            if tal_name in {"husky", "table"}:
                continue

            support_u, support_v = self._support_pixel(det)
            table_xyz = self._pixel_to_world_on_plane(support_u, support_v, camera_pose, self.table_z)
            ground_xyz = self._pixel_to_world_on_plane(support_u, support_v, camera_pose, self.ground_z)
            selected_plane, selected_xyz = self._select_support_plane(
                class_name,
                support_u,
                support_v,
                table_xyz,
                ground_xyz,
                table_bounds,
                table_bbox,
            )
            tal_name = self._resolve_tal_name(class_name, selected_plane)
            if tal_name in {"husky", "table"} or selected_xyz is None:
                continue

            previous = best_by_object.get(tal_name)
            if previous is None or det["confidence"] >= previous["confidence"]:
                best_by_object[tal_name] = {"confidence": det["confidence"], "xyz": selected_xyz}

        return {obj_name: payload["xyz"] for obj_name, payload in best_by_object.items()}

    @staticmethod
    def _support_pixel(det):
        x1, _, x2, y2 = det["xyxy"]
        u, v, _, _ = det["xywh"]
        if det["class"] in {"table", "robot"}:
            return u, v
        return (x1 + x2) / 2.0, y2

    def _estimate_table_info(self, detections, camera_pose, margin=0.25):
        table_candidates = [det for det in detections if "table" in det["class"]]
        if not table_candidates:
            return None
        table = max(table_candidates, key=lambda det: det["confidence"])
        x1, y1, x2, y2 = table["xyxy"]
        points = [
            self._pixel_to_world_on_plane(u, v, camera_pose, self.table_z)
            for u, v in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        ]
        points = [point for point in points if point is not None and self._inside_room(point)]
        if not points:
            return None
        points = np.array(points)
        return {
            "world_bounds": {
                "x_min": float(points[:, 0].min() - margin),
                "x_max": float(points[:, 0].max() + margin),
                "y_min": float(points[:, 1].min() - margin),
                "y_max": float(points[:, 1].max() + margin),
            },
            "image_bbox": [float(x1), float(y1), float(x2), float(y2)],
        }

    def _select_support_plane(self, class_name, support_u, support_v, table_xyz, ground_xyz, table_bounds, table_bbox):
        if class_name in {"robot", "table", "stool"}:
            return "ground", ground_xyz
        if table_bbox is not None and _inside_image_bbox(support_u, support_v, table_bbox, margin_px=20):
            return "table", table_xyz
        if class_name == "bigpallet":
            return "ground", ground_xyz
        if table_xyz is not None and self._inside_room(table_xyz):
            if table_bounds is None or _inside_xy_bounds(table_xyz, table_bounds):
                return "table", table_xyz
        return "ground", ground_xyz

    def _resolve_tal_name(self, class_name, selected_plane):
        # 2026-05-11 修改：不同 TAL2 版本里 cube/bottle 可能叫 cube_red/bottle_red，
        # 也可能叫 cube_gray/bottle_blue；根据当前 config 自动选择，避免 YOLO 结果写不进 graph。
        if class_name == "robot":
            return "husky"
        if class_name == "cube":
            return self._first_existing_object(["cube_gray", "cube_red", "cube"])
        if class_name == "bottle":
            return self._first_existing_object(["bottle_blue", "bottle_red", "bottle"])
        if class_name == "stool":
            return "stool"
        if class_name == "table":
            return "table"
        if class_name in {"smallpallet", "bigpallet"}:
            return "tray" if selected_plane == "table" else "big-tray"
        return class_name

    def _first_existing_object(self, candidates):
        all_objects = set(getattr(self.config, "all_objects", []))
        for candidate in candidates:
            if candidate in all_objects:
                return candidate
        return candidates[0]

    def _pixel_to_world_on_plane(self, u, v, camera_pose, target_z):
        x_c = (u - self.cx) / self.fx
        y_c = -(v - self.cy) / self.fy
        ray_c = np.array([x_c, y_c, -1.0], dtype=np.float64)
        ray_c /= np.linalg.norm(ray_c)
        ray_w = camera_pose[:3, :3] @ ray_c
        cam_origin = camera_pose[:3, 3]
        if abs(ray_w[2]) < 1e-8:
            return None
        t = (target_z - cam_origin[2]) / ray_w[2]
        if t <= 0:
            return None
        return cam_origin + t * ray_w

    def _inside_room(self, point):
        x_min, x_max, y_min, y_max = self.room_bounds
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def _get_usd_attr(prim, attr_name, default):
    attr = prim.GetAttribute(attr_name)
    if attr and attr.HasValue():
        value = attr.Get()
        if value is not None:
            return value
    return default


def _gf_matrix_to_column_pose(matrix):
    raw = np.array([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return raw.T


def _inside_xy_bounds(point, bounds):
    return bounds["x_min"] <= point[0] <= bounds["x_max"] and bounds["y_min"] <= point[1] <= bounds["y_max"]


def _inside_image_bbox(u, v, bbox, margin_px=0):
    x1, y1, x2, y2 = bbox
    return x1 - margin_px <= u <= x2 + margin_px and y1 - margin_px <= v <= y2 + margin_px
