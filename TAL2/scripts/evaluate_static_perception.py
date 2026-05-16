import argparse
import asyncio
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_YOLO_MODEL = "/root/gpufree-data/PRJ/Yolo2/runs/detect/train/weights/best.pt"
DEFAULT_USD_PATH = "/root/Desktop/Collected_exp3/expff.usd"
DEFAULT_CAMERA_PATH = "/World/high"
DEFAULT_ROOM_BOUNDS = (-3.76, 2.09, -3.79, -0.153)
DEFAULT_TABLE_Z = -1.40
DEFAULT_GROUND_Z = -1.93
YOLO_TO_USD_PRIMS = {
    "robot": ["Mobie_grasper2"],
    "cube": ["Cube"],
    "bottle": ["Bottle2"],
    "table": ["table"],
    "stool": ["Stool"],
    "smallpallet": ["SmallPallet"],
    "bigpallet": ["BigPallet"],
}


def log(message):
    print(f"[StaticPerception] {message}", flush=True)


class Vision3DEvaluator:
    YOLO_CLASSES = ["robot", "cube", "bottle", "table", "stool", "smallpallet", "bigpallet"]

    def __init__(
        self,
        dataset_root=None,
        model_path=DEFAULT_YOLO_MODEL,
        *,
        image_width=1024,
        image_height=1024,
        table_z=DEFAULT_TABLE_Z,
        ground_z=DEFAULT_GROUND_Z,
        room_bounds=DEFAULT_ROOM_BOUNDS,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.model = YOLO(model_path)

        self.width = int(image_width)
        self.height = int(image_height)
        self.fx = (24.0 / 20.955) * self.width
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.table_z = float(table_z)
        self.ground_z = float(ground_z)
        self.room_bounds = tuple(float(v) for v in room_bounds)

    def load_camera_pose(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        view_mat = np.array(data["cameraViewTransform"]).reshape(4, 4)
        return np.linalg.inv(view_mat.T)

    def configure_intrinsics_from_usd_camera(self, camera_prim):
        focal = _get_usd_attr(camera_prim, "focalLength", 24.0)
        horizontal_aperture = _get_usd_attr(camera_prim, "horizontalAperture", 20.955)
        vertical_aperture = _get_usd_attr(
            camera_prim,
            "verticalAperture",
            horizontal_aperture * self.height / max(self.width, 1),
        )
        self.fx = float(focal) / float(horizontal_aperture) * self.width
        self.fy = float(focal) / float(vertical_aperture) * self.height
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def pixel_to_world_on_plane(self, u, v, pose, target_z):
        x_c = (u - self.cx) / self.fx
        y_c = -(v - self.cy) / self.fy
        ray_c = np.array([x_c, y_c, -1.0], dtype=np.float64)
        ray_c /= np.linalg.norm(ray_c)

        ray_w = pose[:3, :3] @ ray_c
        cam_origin = pose[:3, 3]

        if abs(ray_w[2]) < 1e-8:
            return None
        t = (target_z - cam_origin[2]) / ray_w[2]
        if t <= 0:
            return None
        return cam_origin + t * ray_w

    def pixel_to_world(self, u, v, pose, class_name):
        plane_z = self.ground_z if class_name in {"robot", "table", "stool"} else self.table_z
        point = self.pixel_to_world_on_plane(u, v, pose, plane_z)
        return np.zeros(3) if point is None else point

    def read_ground_truth(self, frame_id):
        if self.dataset_root is None:
            return []
        gt_path = self.dataset_root / f"bounding_box_3d_{frame_id}.npy"
        label_json_path = self.dataset_root / f"bounding_box_3d_labels_{frame_id}.json"

        with open(label_json_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        gt_data = np.load(gt_path, allow_pickle=True)
        objects = []
        for row in gt_data:
            sem_id = str(row["semanticId"])
            if sem_id not in id_map:
                continue

            raw_name = id_map[sem_id]["class"].lower()
            clean_name = next((yc for yc in self.YOLO_CLASSES if yc in raw_name), None)
            if not clean_name:
                continue

            matrix = np.array(row["transform"]).reshape(4, 4)
            world_pos = matrix[:3, 3]
            if np.allclose(world_pos, 0):
                world_pos = matrix[3, :3]

            objects.append({"name": clean_name, "pos": world_pos})
        return objects

    def detect_image(self, image_bgr, conf=0.35):
        results = self.model.predict(source=image_bgr, conf=conf, verbose=False)[0]
        model_names = getattr(self.model, "names", None) or {}
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u, v, w, h = box.xywh[0].tolist()
            cls_id = int(box.cls[0].item())
            class_name = model_names.get(cls_id, self.YOLO_CLASSES[cls_id]).lower()
            confidence = float(box.conf[0].item())
            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [u, v, w, h],
                }
            )
        return detections

    def estimate_positions(self, detections, camera_pose):
        table_info = self._estimate_table_info(detections, camera_pose)
        table_bounds = None if table_info is None else table_info["world_bounds"]
        table_bbox = None if table_info is None else table_info["image_bbox"]
        estimates = []

        for det in detections:
            class_name = det["class"]
            u, v, _, h = det["xywh"]
            x1, y1, x2, y2 = det["xyxy"]

            # Bottom-center is usually a better support point than box center.
            support_u = (x1 + x2) / 2.0
            support_v = y2
            if class_name in {"table", "robot"}:
                support_u, support_v = u, v

            table_xyz = self.pixel_to_world_on_plane(support_u, support_v, camera_pose, self.table_z)
            ground_xyz = self.pixel_to_world_on_plane(support_u, support_v, camera_pose, self.ground_z)
            selected_plane, selected_xyz = self._select_support_plane(
                class_name,
                support_u,
                support_v,
                table_xyz,
                ground_xyz,
                table_bounds,
                table_bbox,
            )
            resolved_class = self._resolve_scene_class(class_name, selected_plane)

            estimates.append(
                {
                    "class": class_name,
                    "resolved_class": resolved_class,
                    "confidence": det["confidence"],
                    "bbox_xyxy": det["xyxy"],
                    "support_pixel": [support_u, support_v],
                    "selected_plane": selected_plane,
                    "selected_xyz": selected_xyz,
                    "table_xyz": table_xyz,
                    "ground_xyz": ground_xyz,
                    "bbox_height_px": h,
                }
            )
        return estimates, table_bounds

    def _estimate_table_info(self, detections, camera_pose, margin=0.25):
        table_candidates = [det for det in detections if "table" in det["class"]]
        if not table_candidates:
            return None

        table = max(table_candidates, key=lambda det: det["confidence"])
        x1, y1, x2, y2 = table["xyxy"]
        pixels = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        points = [
            self.pixel_to_world_on_plane(u, v, camera_pose, self.table_z)
            for u, v in pixels
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

    def _select_support_plane(
        self,
        class_name,
        support_u,
        support_v,
        table_xyz,
        ground_xyz,
        table_bounds,
        table_bbox,
    ):
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

    def _resolve_scene_class(self, class_name, selected_plane):
        if class_name in {"bigpallet", "smallpallet"}:
            return "smallpallet" if selected_plane == "table" else "bigpallet"
        return class_name

    def _inside_room(self, point):
        x_min, x_max, y_min, y_max = self.room_bounds
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

    def run_dataset(self, num_frames=20, conf=0.5):
        if self.dataset_root is None:
            raise ValueError("dataset_root is required for dataset mode")

        all_errors = []
        for i in range(num_frames):
            fid = f"{i:04d}"
            img_path = self.dataset_root / f"rgb_{fid}.png"
            if not img_path.exists():
                continue

            try:
                pose = self.load_camera_pose(self.dataset_root / f"camera_params_{fid}.json")
                gts = self.read_ground_truth(fid)
                image_bgr = cv2.imread(str(img_path))
                detections = self.detect_image(image_bgr, conf=conf)
                estimates, _ = self.estimate_positions(detections, pose)

                print(f"\n--- Frame {fid} ---")
                print(f"{'Class':<12} | {'Pred XYZ':<24} | {'GT XYZ':<24} | {'Err(cm)'}")
                for estimate in estimates:
                    pos = estimate["selected_xyz"]
                    if pos is None:
                        continue
                    best_err = 999.0
                    best_gt = [0, 0, 0]
                    for gt in gts:
                        if gt["name"] == estimate["class"]:
                            err = np.linalg.norm(pos - gt["pos"])
                            if err < best_err:
                                best_err = err
                                best_gt = gt["pos"]

                    if best_err < 9:
                        print(
                            f"{estimate['class']:<12} | {np.round(pos, 3)} | "
                            f"{np.round(best_gt, 3)} | {best_err * 100:.2f}cm"
                        )
                        all_errors.append(best_err)
            except Exception as exc:
                print(f"Error in Frame {fid}: {exc}")

        if all_errors:
            print(f"\nFinal Overall Mean Error: {np.mean(all_errors) * 100:.2f} cm")


def _get_usd_attr(prim, attr_name, default):
    attr = prim.GetAttribute(attr_name)
    if attr and attr.HasValue():
        value = attr.Get()
        if value is not None:
            return value
    return default


def _inside_xy_bounds(point, bounds):
    return (
        bounds["x_min"] <= point[0] <= bounds["x_max"]
        and bounds["y_min"] <= point[1] <= bounds["y_max"]
    )


def _inside_image_bbox(u, v, bbox, margin_px=0):
    x1, y1, x2, y2 = bbox
    return (
        x1 - margin_px <= u <= x2 + margin_px
        and y1 - margin_px <= v <= y2 + margin_px
    )


def _gf_matrix_to_column_pose(matrix):
    raw = np.array([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return raw.T


def _read_usd_ground_truth(stage, UsdGeom):
    xform_cache = UsdGeom.XformCache()
    ground_truth = {}
    for class_name, usd_names in YOLO_TO_USD_PRIMS.items():
        class_entries = []
        for usd_name in usd_names:
            prim_path = f"/World/{usd_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            pose = _gf_matrix_to_column_pose(xform_cache.GetLocalToWorldTransform(prim))
            class_entries.append({"prim_path": prim_path, "xyz": pose[:3, 3]})
        if class_entries:
            ground_truth[class_name] = class_entries
    return ground_truth


def _nearest_ground_truth(class_name, selected_xyz, ground_truth):
    candidates = ground_truth.get(class_name, [])
    if selected_xyz is None or not candidates:
        return None, None
    selected_xyz = np.asarray(selected_xyz, dtype=float)
    best = min(candidates, key=lambda item: np.linalg.norm(selected_xyz - item["xyz"]))
    return best, float(np.linalg.norm(selected_xyz - best["xyz"]))


def _frame_image_from_current_frame(camera):
    frame = camera.get_current_frame(clone=True)
    log(f"Camera current frame keys: {sorted(frame.keys())}")
    image = frame.get("rgba")
    if image is None:
        image = frame.get("rgb")
    if image is not None:
        log(
            "Camera current frame image: "
            f"type={type(image).__name__}, shape={getattr(image, 'shape', None)}, "
            f"dtype={getattr(image, 'dtype', None)}, size={getattr(image, 'size', None)}"
        )
    return image


def _read_camera_rgba(camera, update_stage, simulation_app, warmup_steps):
    try:
        image_rgba = camera.get_rgba()
        if image_rgba is not None and getattr(image_rgba, "size", 0) != 0:
            return image_rgba
        log("Camera.get_rgba() did not return data; trying get_current_frame() fallback")
    except Exception as exc:
        log(f"Camera.get_rgba() failed: {exc}")
        log(traceback.format_exc())

    try:
        camera.add_rgb_to_frame()
    except Exception as exc:
        log(f"camera.add_rgb_to_frame() warning: {exc}")

    for _ in range(max(warmup_steps, 1)):
        update_stage()
        simulation_app.update()

    image_rgba = None
    for retry_idx in range(10):
        image_rgba = _frame_image_from_current_frame(camera)
        if image_rgba is not None and getattr(image_rgba, "size", 0) != 0:
            return image_rgba
        log(f"RGB frame is not ready yet; retry {retry_idx + 1}/10")
        for _ in range(5):
            update_stage()
            simulation_app.update()
    return image_rgba


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


def _read_replicator_rgb(camera_path, width, height, warmup_steps):
    import omni.replicator.core as rep

    log("Trying direct Replicator render product capture")
    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(camera_path, resolution=(width, height))
    log(f"Replicator render product: {getattr(render_product, 'path', render_product)}")
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)

    try:
        image = None
        for retry_idx in range(10):
            simulation_app = getattr(_read_replicator_rgb, "_simulation_app", None)
            update_stage = getattr(_read_replicator_rgb, "_update_stage", None)
            try:
                rep.orchestrator.step(rt_subframes=max(warmup_steps, 1))
            except TypeError:
                rep.orchestrator.step()
            for _ in range(3):
                if update_stage is not None:
                    update_stage()
                if simulation_app is not None:
                    simulation_app.update()
            image = rgb_annotator.get_data()
            log(
                "Replicator rgb: "
                f"type={type(image).__name__}, shape={getattr(image, 'shape', None)}, "
                f"dtype={getattr(image, 'dtype', None)}, size={getattr(image, 'size', None)}"
            )
            if image is not None and getattr(image, "size", 0) != 0:
                return np.asarray(image)
            log(f"Replicator RGB frame is not ready yet; retry {retry_idx + 1}/10")
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


def run_isaac_scene(args):
    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    try:
        try:
            from isaacsim.core.utils.stage import get_current_stage, open_stage, update_stage
            from isaacsim.sensors.camera import Camera
        except ModuleNotFoundError:
            from omni.isaac.core.utils.stage import get_current_stage, open_stage, update_stage
            from omni.isaac.sensor import Camera
        import omni.timeline
        from pxr import UsdGeom

        log(f"Opening USD stage: {args.usd_path}")
        if not open_stage(args.usd_path):
            raise RuntimeError(f"Failed to open USD stage: {args.usd_path}")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(args.warmup_steps):
            update_stage()
            simulation_app.update()

        stage = get_current_stage()
        camera_path = args.camera_path
        if not camera_path.startswith("/"):
            camera_path = f"/World/{camera_path}"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"Camera prim not found: {camera_path}")
        log(f"Using camera prim: {camera_path}")

        evaluator = Vision3DEvaluator(
            model_path=args.model_path,
            image_width=args.width,
            image_height=args.height,
            table_z=args.table_z,
            ground_z=args.ground_z,
            room_bounds=args.room_bounds,
        )
        evaluator.configure_intrinsics_from_usd_camera(camera_prim)
        log(
            "Camera intrinsics: "
            f"fx={evaluator.fx:.3f}, fy={evaluator.fy:.3f}, "
            f"cx={evaluator.cx:.1f}, cy={evaluator.cy:.1f}"
        )

        camera = Camera(prim_path=camera_path, resolution=(args.width, args.height))
        camera.initialize()
        for _ in range(args.warmup_steps):
            update_stage()
            simulation_app.update()

        log("Reading RGB frame from Isaac camera")
        image_rgba = _read_camera_rgba(camera, update_stage, simulation_app, args.warmup_steps)
        if image_rgba is None or getattr(image_rgba, "size", 0) == 0:
            _read_replicator_rgb._simulation_app = simulation_app
            _read_replicator_rgb._update_stage = update_stage
            image_rgba = _read_replicator_rgb(camera_path, args.width, args.height, args.warmup_steps)
        if image_rgba is None:
            raise RuntimeError("Isaac camera returned no rgb/rgba frame")
        if image_rgba.size == 0:
            raise RuntimeError("Isaac camera returned an empty image")
        log(f"Camera frame shape={image_rgba.shape}, dtype={image_rgba.dtype}")

        image_rgb = image_rgba[:, :, :3]
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if args.output_image:
            output_path = Path(args.output_image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(output_path), image_bgr)
            log(f"Saved camera image: {output_path} ok={ok}")
            if not ok:
                raise RuntimeError(f"Failed to write camera image: {output_path}")

        camera_pose = _gf_matrix_to_column_pose(
            UsdGeom.XformCache().GetLocalToWorldTransform(camera_prim)
        )
        log(f"Running YOLO: {args.model_path}")
        detections = evaluator.detect_image(image_bgr, conf=args.conf)
        log(f"YOLO detections: {len(detections)}")
        estimates, table_bounds = evaluator.estimate_positions(detections, camera_pose)
        ground_truth = _read_usd_ground_truth(stage, UsdGeom)

        print(f"USD: {args.usd_path}", flush=True)
        print(f"Camera: {camera_path}", flush=True)
        print(f"YOLO model: {args.model_path}", flush=True)
        print(f"Table z: {args.table_z:.3f}, ground z: {args.ground_z:.3f}", flush=True)
        print(f"Table XY bounds from detection: {table_bounds}", flush=True)
        print("", flush=True)
        print(
            f"{'Class':<12} | {'Conf':<5} | {'Plane':<6} | {'Selected XYZ':<28} | "
            f"{'GT XYZ':<28} | {'Err cm':<7} | {'GT Prim':<28} | "
            f"{'Table XYZ':<28} | {'Ground XYZ'}",
            flush=True,
        )
        for item in estimates:
            gt, err_m = _nearest_ground_truth(item["resolved_class"], item["selected_xyz"], ground_truth)
            gt_xyz = None if gt is None else gt["xyz"]
            err_cm = "N/A" if err_m is None else f"{err_m * 100:.1f}"
            gt_prim = "N/A" if gt is None else gt["prim_path"]
            class_label = item["class"]
            if item["resolved_class"] != item["class"]:
                class_label = f"{item['class']}->{item['resolved_class']}"
            print(
                f"{class_label:<12} | {item['confidence']:<5.2f} | {item['selected_plane']:<6} | "
                f"{_fmt_vec(item['selected_xyz']):<28} | {_fmt_vec(gt_xyz):<28} | "
                f"{err_cm:<7} | {gt_prim:<28} | {_fmt_vec(item['table_xyz']):<28} | "
                f"{_fmt_vec(item['ground_xyz'])}",
                flush=True,
            )
        if not estimates:
            log("No detections were converted to coordinates. Try lowering --conf or inspect the saved image.")
    except Exception as exc:
        log(f"ERROR: {exc}")
        log(traceback.format_exc())
        raise
    finally:
        try:
            timeline.stop()
        except Exception:
            pass
        log("Closing SimulationApp")
        simulation_app.close()


def _fmt_vec(value):
    if value is None:
        return "None"
    return str(np.round(np.asarray(value, dtype=float), 3).tolist())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO perception and estimate object 3D positions.")
    parser.add_argument("--mode", choices=["dataset", "isaac"], default="isaac")
    parser.add_argument("--dataset-root", default="/root/gpufree-data/YOLO_Dataset_360")
    parser.add_argument("--model-path", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.35)

    parser.add_argument("--usd-path", default=DEFAULT_USD_PATH)
    parser.add_argument("--camera-path", default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z)
    parser.add_argument("--ground-z", type=float, default=DEFAULT_GROUND_Z)
    parser.add_argument("--room-bounds", type=float, nargs=4, default=DEFAULT_ROOM_BOUNDS)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-image", default="")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "dataset":
        evaluator = Vision3DEvaluator(
            dataset_root=cli_args.dataset_root,
            model_path=cli_args.model_path,
            image_width=cli_args.width,
            image_height=cli_args.height,
            table_z=cli_args.table_z,
            ground_z=cli_args.ground_z,
            room_bounds=cli_args.room_bounds,
        )
        evaluator.run_dataset(cli_args.num_frames, conf=cli_args.conf)
    else:
        run_isaac_scene(cli_args)
