from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import cv2
import numpy as np


parser = argparse.ArgumentParser(description="Isaac Sim TAL + OpenPI closed-loop controller")
parser.add_argument("--prompt", type=str, default="pick up the block", help="The language instruction for the robot")
parser.add_argument("--server-host", type=str, default="127.0.0.1", help="OpenPI policy server host")
parser.add_argument("--server-port", type=int, default=8000, help="OpenPI policy server port")
parser.add_argument("--tal-root", type=str, required=True, help="Path to TAL2 repo root")
parser.add_argument("--qwen-model", type=str, default="qwen3-max", help="DashScope model name used by TAL")
parser.add_argument("--qwen-api-key-env", type=str, default="DASHSCOPE_API_KEY", help="Env var storing DashScope key")
parser.add_argument("--manual-scene-graph-json", type=str, default="", help="Optional JSON file path for scene graph")
parser.add_argument("--replan-every-n-steps", type=int, default=1, help="Replan every N control steps")
parser.add_argument("--max-steps", type=int, default=-1, help="Maximum control loop steps; -1 means unlimited")
parser.add_argument(
    "--tal-world-state-name",
    type=str,
    default="Initialize",
    help='Initial TAL scene graph state token for debug, for example "Initialize"',
)
parser.add_argument("--nav-control-dt", type=float, default=0.05, help="Navigation bridge/control period in seconds")
parser.add_argument("--nav-goal-timeout-sec", type=float, default=120.0, help="Timeout for a single Nav2 goal")
parser.add_argument(
    "--nav-server-timeout-sec",
    type=float,
    default=45.0,
    help="Timeout for bt_navigator / NavigateToPose server to become ready before sending a goal",
)
parser.add_argument(
    "--nav-warmup-sec",
    type=float,
    default=4.0,
    help="Seconds to advance simulation after starting the Nav2 bridge so /clock, /odom, and /tf are live",
)
parser.add_argument(
    "--wheel-self-test",
    action="store_true",
    default=False,
    help="Run a direct wheel-velocity self test and exit before TAL/OpenPI/Nav2 closed-loop control",
)
parser.add_argument(
    "--wheel-self-test-duration-sec",
    type=float,
    default=5.0,
    help="Duration for the direct wheel-velocity self test",
)
parser.add_argument(
    "--wheel-self-test-left-rad-s",
    type=float,
    default=2.0,
    help="Left wheel angular velocity used in wheel self test",
)
parser.add_argument(
    "--wheel-self-test-right-rad-s",
    type=float,
    default=2.0,
    help="Right wheel angular velocity used in wheel self test",
)
parser.add_argument("--headless", action="store_true", default=False, help="Run Isaac Sim in headless mode")
args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args


CAMERA_HIGH_PATH = os.environ.get("TAL_ONLINE_HIGH_CAMERA_PATH", "/World/Mobie_grasper2/high")
CAMERA_WRIST_PATH = "/World/Mobie_grasper2/firefighter/joint6/wrist"
# 2026-06-08 修改：high 第一人称相机已从 /World/high 移到 /World/Mobie_grasper2/high。
# OpenPI/VLA 默认读新的车载 high；TAL/YOLO 未单独指定时也复用这一路。
CAMERA_TAL_PATH = os.environ.get("TAL_ONLINE_TAL_CAMERA_PATH", os.environ.get("TAL_YOLO_CAMERA_PATH", CAMERA_HIGH_PATH))
ROBOT_START_WORLD_POSITION = np.array([-0.13648, -1.41058, -1.76984], dtype=np.float32)
TRAIN_INIT_STATE = np.array(
    [-0.12466581, -0.15327631, 1.2, -0.1757595, 1.5070096, -0.320009, 0.13824108],
    dtype=np.float32,
)
JOINT_NAMES_IN_ORDER = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "finger_joint",
]
NAV_MAP_YAML_PATH = Path(
    os.environ.get(
        "TAL_NAV_MAP_YAML",
        "/root/gpufree-data/code/tal-vla/robot_ws/src/robot_navigation/maps/expff_map.yaml",
    )
)


def _should_move_robot_root() -> bool:
    raw = os.environ.get("TAL_ONLINE_MOVE_ROBOT_ROOT")
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _reinitialize_robot_if_possible(robot: Any) -> None:
    if hasattr(robot, "initialize"):
        try:
            robot.initialize()
        except Exception:
            pass
    if hasattr(robot, "post_reset"):
        try:
            robot.post_reset()
        except Exception:
            pass


def _initialize_robot_handles_if_needed(robot: Any, world: Any, *, headless: bool, root_pose_guard: Any | None = None) -> None:
    init_steps = max(int(os.environ.get("TAL_ONLINE_ARTICULATION_INIT_STEPS", "30")), 1)
    for _ in range(init_steps):
        _step_world_with_root_guard(world, render=not headless, root_pose_guard=root_pose_guard)
        time.sleep(0.02)
        if hasattr(robot, "initialize"):
            try:
                robot.initialize()
            except Exception:
                pass
        dof_names = getattr(robot, "dof_names", None)
        if dof_names is not None:
            return

    if not _allow_reset_fallback():
        raise RuntimeError(
            "Articulation handle is still not ready after non-reset initialization; "
            "set TAL_ONLINE_ALLOW_RESET_FALLBACK=1 only if you accept a reset fallback."
        )

    print("[InitRobot] non-reset initialization failed; using reset fallback", flush=True)
    world.reset()
    for _ in range(init_steps):
        _step_world_with_root_guard(world, render=not headless, root_pose_guard=root_pose_guard)
        time.sleep(0.02)
        if hasattr(robot, "initialize"):
            try:
                robot.initialize()
            except Exception:
                pass
        dof_names = getattr(robot, "dof_names", None)
        if dof_names is not None:
            return

    raise RuntimeError("Articulation handle is still not ready even after reset fallback")


def _should_warmup_robot() -> bool:
    raw = os.environ.get("TAL_ONLINE_WARMUP_ROBOT")
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _should_reset_world() -> bool:
    raw = os.environ.get("TAL_ONLINE_RESET_WORLD")
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _should_reinitialize_robot() -> bool:
    raw = os.environ.get("TAL_ONLINE_REINITIALIZE_ROBOT")
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _allow_reset_fallback() -> bool:
    raw = os.environ.get("TAL_ONLINE_ALLOW_RESET_FALLBACK")
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _render_after_root_enforce() -> bool:
    raw = os.environ.get("TAL_NAV_RENDER_AFTER_ROOT_ENFORCE")
    if raw is None:
        return True
    return raw.lower() in {"1", "true", "yes", "on"}


def _render_world_if_possible(world: Any) -> None:
    if hasattr(world, "render"):
        try:
            world.render()
            return
        except Exception:
            pass
    try:
        import omni.kit.app  # type: ignore

        omni.kit.app.get_app().update()
    except Exception:
        pass


def _step_world_with_root_guard(world: Any, *, render: bool, root_pose_guard: Any | None = None) -> None:
    render_after_enforce = render and root_pose_guard is not None and _render_after_root_enforce()
    world.step(render=False if render_after_enforce else render)
    if root_pose_guard is not None:
        root_pose_guard.enforce_after_step()
    if render_after_enforce:
        _render_world_if_possible(world)


def _get_initial_robot_state() -> np.ndarray:
    raw = os.environ.get("TAL_ONLINE_INITIAL_STATE", "").strip()
    if not raw:
        return TRAIN_INIT_STATE.copy()
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != len(JOINT_NAMES_IN_ORDER):
        raise ValueError(
            f"TAL_ONLINE_INITIAL_STATE expects {len(JOINT_NAMES_IN_ORDER)} values, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float32)


def _capture_current_robot_pose_from_stage(robot_prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        import omni.usd  # type: ignore
    except Exception:
        return None, None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None

    def _find_joint_prim(joint_name: str) -> Any | None:
        candidate_paths = [
            f"{robot_prim_path}/firefighter/joints/{joint_name}",
            f"{robot_prim_path}/joints/{joint_name}",
        ]
        for prim_path in candidate_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return prim
        for prim in stage.Traverse():
            try:
                prim_path = str(prim.GetPath())
                prim_name = prim.GetName()
            except Exception:
                continue
            if prim_name == joint_name and prim_path.startswith(robot_prim_path):
                return prim
        return None

    def _read_joint_value(prim: Any) -> float | None:
        attr_names = [
            "drive:angular:physics:targetPosition",
            "state:angular:physics:position",
            "drive:linear:physics:targetPosition",
            "state:linear:physics:position",
        ]
        for attr_name in attr_names:
            try:
                attr = prim.GetAttribute(attr_name)
            except Exception:
                attr = None
            if not attr:
                continue
            try:
                value = attr.Get()
            except Exception:
                value = None
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    ordered = []
    for name in JOINT_NAMES_IN_ORDER:
        prim = _find_joint_prim(name)
        if prim is None:
            return None, None
        value = _read_joint_value(prim)
        if value is None:
            return None, None
        ordered.append(value)

    return np.asarray(ordered, dtype=np.float32), None


def _capture_current_robot_pose(robot: Any, robot_prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        dof_names = getattr(robot, "dof_names", None)
        joint_pos = robot.get_joint_positions() if hasattr(robot, "get_joint_positions") else None
    except Exception:
        dof_names = None
        joint_pos = None
    if dof_names is None or joint_pos is None:
        return _capture_current_robot_pose_from_stage(robot_prim_path)
    ordered = []
    target_indices = []
    for name in JOINT_NAMES_IN_ORDER:
        if name not in dof_names:
            return _capture_current_robot_pose_from_stage(robot_prim_path)
        idx = dof_names.index(name)
        target_indices.append(idx)
        ordered.append(joint_pos[idx])
    return np.asarray(ordered, dtype=np.float32), np.asarray(target_indices, dtype=np.int32)


def _restore_robot_pose_if_available(
    robot: Any,
    world: Any,
    pose_state: np.ndarray | None,
    pose_indices: np.ndarray | None,
    ArticulationAction: Any,
    *,
    headless: bool,
    root_pose_guard: Any | None = None,
) -> None:
    if pose_state is None:
        return
    if pose_indices is None:
        dof_names = getattr(robot, "dof_names", None)
        if dof_names is None:
            return
        remapped_indices = []
        for name in JOINT_NAMES_IN_ORDER:
            if name not in dof_names:
                return
            remapped_indices.append(dof_names.index(name))
        pose_indices = np.asarray(remapped_indices, dtype=np.int32)
    restore_steps = max(int(os.environ.get("TAL_ONLINE_RESTORE_POSE_STEPS", "60")), 1)
    restore_action = ArticulationAction(
        joint_positions=np.asarray(pose_state, dtype=np.float32),
        joint_indices=np.asarray(pose_indices, dtype=np.int32),
    )
    for _ in range(restore_steps):
        robot.apply_action(restore_action)
        _step_world_with_root_guard(world, render=not headless, root_pose_guard=root_pose_guard)


@dataclasses.dataclass
class TALPlanResult:
    status: str
    first_action_text: str | None
    predicted_actions: list[Any]
    current_scene_graph_json: dict[str, Any] | None = None
    goal_scene_graph_json: dict[str, Any] | None = None
    error: str | None = None


@dataclasses.dataclass
class NavigationGoal:
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"


@dataclasses.dataclass
class NavOccupancyMap:
    resolution: float
    origin_x: float
    origin_y: float
    image: np.ndarray

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        height, width = self.image.shape
        ix = int((x - self.origin_x) / self.resolution)
        iy = height - 1 - int((y - self.origin_y) / self.resolution)
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            return None
        return ix, iy

    def has_clearance(self, x: float, y: float, radius_m: float) -> bool:
        grid = self.world_to_grid(x, y)
        if grid is None:
            return False
        ix, iy = grid
        radius_cells = max(int(math.ceil(radius_m / self.resolution)), 0)
        height, width = self.image.shape
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                x2 = ix + dx
                y2 = iy + dy
                if x2 < 0 or y2 < 0 or x2 >= width or y2 >= height:
                    return False
                if int(self.image[y2, x2]) < 250:
                    return False
        return True


@dataclasses.dataclass
class PendingNavigation:
    goal: NavigationGoal
    accepted: bool = False
    success: bool = False
    status: int | None = None
    error: str | None = None
    done_event: threading.Event = dataclasses.field(default_factory=threading.Event)


@dataclasses.dataclass
class ParsedTALSubtask:
    name: str
    args: list[str]
    text: str | None = None
    raw: Any | None = None

    @property
    def is_navigation(self) -> bool:
        return self.name.lower() == "moveto" and len(self.args) >= 1


def _normalize_tal_arg(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("\"'")


def parse_tal_subtask(action: Any) -> ParsedTALSubtask | None:
    if action is None:
        return None
    if isinstance(action, Mapping):
        name = str(action.get("name", "")).strip()
        args_value = action.get("args", [])
        if not isinstance(args_value, list):
            args_value = [args_value]
        args = [_normalize_tal_arg(arg) for arg in args_value if _normalize_tal_arg(arg)]
        if not name:
            return None
        return ParsedTALSubtask(name=name, args=args, text=format_tal_action(action), raw=action)

    action_text = format_tal_action(action)
    if not action_text:
        return None

    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", action_text)
    if match:
        name = match.group(1)
        args_blob = match.group(2).strip()
        args = [_normalize_tal_arg(item) for item in args_blob.split(",") if _normalize_tal_arg(item)]
        return ParsedTALSubtask(name=name, args=args, text=action_text, raw=action)

    return ParsedTALSubtask(name=action_text.strip(), args=[], text=action_text, raw=action)


def derive_executable_subtask(parsed_subtask: ParsedTALSubtask | None) -> ParsedTALSubtask | None:
    if parsed_subtask is None:
        return None
    if parsed_subtask.name.lower() in {"picknplaceaonb", "pushto"} and parsed_subtask.args:
        nav_target = parsed_subtask.args[0]
        return ParsedTALSubtask(
            name="moveTo",
            args=[nav_target],
            text=f"moveTo({nav_target})",
            raw={
                "name": "moveTo",
                "args": [nav_target],
                "derived_from": parsed_subtask.raw if parsed_subtask.raw is not None else parsed_subtask.text,
            },
        )
    return parsed_subtask


def quaternion_to_yaw(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> float:
    if quaternion is None:
        return 0.0
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return 0.0
    w, x, y, z = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def quaternion_to_rpy(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> tuple[float, float, float]:
    if quaternion is None:
        return 0.0, 0.0, 0.0
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return 0.0, 0.0, 0.0
    w, x, y, z = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rpy_to_quaternion(roll, pitch, yaw)


def quaternion_wxyz_to_rpy(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> tuple[float, float, float]:
    if quaternion is None:
        return 0.0, 0.0, 0.0
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return 0.0, 0.0, 0.0
    return quaternion_to_rpy(q)


def quaternion_wxyz_to_yaw(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> float:
    return quaternion_wxyz_to_rpy(quaternion)[2]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def navigation_goal_is_close_enough(
    robot_root_controller: "RobotRootPoseController",
    goal: NavigationGoal,
    *,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> bool:
    position, orientation = robot_root_controller.get_world_pose()
    if position is None:
        return False
    position = np.asarray(position, dtype=np.float32)
    distance_xy = float(np.linalg.norm(position[:2] - np.array([goal.x, goal.y], dtype=np.float32)))
    current_yaw = quaternion_wxyz_to_yaw(orientation) if orientation is not None else 0.0
    yaw_error = abs(normalize_angle(current_yaw - goal.yaw))
    return distance_xy <= position_tolerance_m and yaw_error <= yaw_tolerance_rad


@dataclasses.dataclass
class TALControllerConfig:
    tal_root: str
    qwen_model: str = "qwen3-max"
    qwen_api_key_env: str = "DASHSCOPE_API_KEY"
    candidate_action_num: int = 20
    select_from_candidate: int = 10
    max_planning_steps: int = 60
    headless: bool = False


@dataclasses.dataclass
class TALRuntimeContext:
    tal_root: Path
    sim_env_config: Any
    planner_env_config: Any
    approx: Any
    isaac_env: Any
    scene_graph_translator: Any
    plan_with_natural_language_instruction: Any
    scene_graph_json_to_dgl: Any
    model_action: Any
    model_action_effect: Any
    action_effect_features: Any
    simulation_app: Any
    qwen_model: str
    qwen_api_key_env: str
    candidate_action_num: int
    select_from_candidate: int
    max_planning_steps: int

    def close(self) -> None:
        try:
            self.approx.close_backend()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to close TAL planner backend cleanly: {exc}")
        try:
            self.isaac_env.destroy()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to close TAL Isaac backend cleanly: {exc}")


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_fused_prompt(original_instruction: str, tal_first_action: str | None) -> str:
    if not tal_first_action:
        return original_instruction
    return f"User task: {original_instruction.strip()}.\nCurrent subtask: {tal_first_action.strip()}."


def format_tal_action(action: Any) -> str | None:
    if action is None:
        return None
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, Mapping):
        name = str(action.get("name", "")).strip()
        args = action.get("args", [])
        if not isinstance(args, list):
            args = [args]
        args_text = ", ".join(str(arg) for arg in args if str(arg).strip())
        if name and args_text:
            return f"{name}({args_text})"
        if name:
            return name
        return json.dumps(action, ensure_ascii=False)
    if isinstance(action, (list, tuple)):
        return ", ".join(str(item) for item in action)
    return str(action).strip()


def _to_abs_repo_path(repo_root: Path, maybe_relative: str) -> str:
    path = Path(maybe_relative)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _build_env_config(tal_root: Path, init_args: Any, EnvironmentConfig: Any, *, policy_backend: str, qwen_model: str, qwen_api_key_env: str) -> Any:
    with pushd(tal_root):
        tal_args = init_args()
        tal_args.exec_type = "policy"
        tal_args.policy_backend = policy_backend
        tal_args.qwen_model = qwen_model

        import torch

        tal_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tal_args.qwen_api_key = os.getenv(qwen_api_key_env) if qwen_api_key_env else None
        if getattr(tal_args, "data_dir", None):
            tal_args.data_dir = _to_abs_repo_path(tal_root, tal_args.data_dir)

        env_config = EnvironmentConfig(tal_args)

    env_config.MODEL_SAVE_PATH = _to_abs_repo_path(tal_root, env_config.MODEL_SAVE_PATH)
    env_config.Aall_path = _to_abs_repo_path(tal_root, env_config.Aall_path)
    env_config.all_possible_actions_path = _to_abs_repo_path(tal_root, env_config.all_possible_actions_path)
    return env_config


def _load_required_model(env_config: Any, get_model: Any, load_model: Any, model_name: str) -> Any:
    model = get_model(env_config, model_name, env_config.features_dim, env_config.num_objects)
    seq_prefix = "Seq_" if env_config.training == "gcn_seq" else ""
    stable_ckpt = Path(env_config.MODEL_SAVE_PATH) / f"{seq_prefix}{model.name}_Trained.ckpt"
    ckpt_path = stable_ckpt if stable_ckpt.exists() else None
    if ckpt_path is None:
        model_dir = Path(env_config.MODEL_SAVE_PATH)
        best_epoch = -1
        for filename in model_dir.iterdir():
            if not filename.name.startswith(seq_prefix + model.name + "_") or filename.suffix != ".ckpt":
                continue
            try:
                epoch = int(filename.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if epoch > best_epoch:
                best_epoch = epoch
                ckpt_path = filename
    if ckpt_path is None:
        raise FileNotFoundError(f"Could not find checkpoint for TAL model {model.name}")
    model, _, _, _ = load_model(env_config, seq_prefix + model.name + "_Trained", model, file_path=str(ckpt_path))
    return model.to(env_config.device)


def initialize_tal_runtime(config: TALControllerConfig) -> TALRuntimeContext:
    tal_root = Path(config.tal_root).resolve()
    if not (tal_root / "src").exists():
        raise FileNotFoundError(f"Invalid TAL root: {tal_root}")

    os.environ["TAL_ISAAC_HEADLESS"] = "1" if config.headless else "0"

    if str(tal_root) not in sys.path:
        sys.path.insert(0, str(tal_root))

    tal_config_module = __import__("src.config.config", fromlist=["init_args"])
    env_constants_module = __import__("src.envs.CONSTANTS", fromlist=["EnvironmentConfig"])
    planning_module = __import__("src.tal.utils_planning", fromlist=["plan_with_natural_language_instruction"])
    training_module = __import__("src.tal.utils_training", fromlist=["get_model", "load_model"])
    translator_module = __import__(
        "src.tal.scene_graph_translator",
        fromlist=["scene_graph_json_to_dgl", "datapoint_to_scene_graph_json"],
    )
    approx_module = __import__("src.envs.approx", fromlist=["initPolicy", "close_backend"])
    isaac_env_module = __import__("src.envs.isaac_env", fromlist=["start", "getDatapoint", "simulation_app"])

    init_args = tal_config_module.init_args
    EnvironmentConfig = env_constants_module.EnvironmentConfig
    plan_with_natural_language_instruction = planning_module.plan_with_natural_language_instruction
    scene_graph_json_to_dgl = translator_module.scene_graph_json_to_dgl
    get_model = training_module.get_model
    load_model = training_module.load_model

    sim_env_config = _build_env_config(
        tal_root,
        init_args,
        EnvironmentConfig,
        policy_backend="isaaclab",
        qwen_model=config.qwen_model,
        qwen_api_key_env=config.qwen_api_key_env,
    )
    planner_env_config = _build_env_config(
        tal_root,
        init_args,
        EnvironmentConfig,
        policy_backend="symbolic",
        qwen_model=config.qwen_model,
        qwen_api_key_env=config.qwen_api_key_env,
    )

    import pickle

    model_action_effect = _load_required_model(planner_env_config, get_model, load_model, "AFE")
    model_action = _load_required_model(planner_env_config, get_model, load_model, "APN")
    features_save_path = Path(planner_env_config.MODEL_SAVE_PATH) / "action_effect_features_avg.pkl"
    with features_save_path.open("rb") as file_obj:
        action_effect_features = pickle.load(file_obj)

    world_num = 0
    graph_world_name = getattr(sim_env_config, "graph_world_name", "")
    digits = "".join(ch for ch in str(graph_world_name) if ch.isdigit())
    if digits:
        world_num = int(digits)

    approx_module.initPolicy(
        sim_env_config,
        sim_env_config.domain,
        goal_json=None,
        world_num=world_num,
        SET_GAOL_JSON=False,
    )

    return TALRuntimeContext(
        tal_root=tal_root,
        sim_env_config=sim_env_config,
        planner_env_config=planner_env_config,
        approx=approx_module,
        isaac_env=isaac_env_module,
        scene_graph_translator=translator_module,
        plan_with_natural_language_instruction=plan_with_natural_language_instruction,
        scene_graph_json_to_dgl=scene_graph_json_to_dgl,
        model_action=model_action,
        model_action_effect=model_action_effect,
        action_effect_features=action_effect_features,
        simulation_app=isaac_env_module.simulation_app,
        qwen_model=config.qwen_model,
        qwen_api_key_env=config.qwen_api_key_env,
        candidate_action_num=config.candidate_action_num,
        select_from_candidate=config.select_from_candidate,
        max_planning_steps=config.max_planning_steps,
    )


class TALSceneGraphProvider:
    def __init__(self, runtime_ctx: TALRuntimeContext):
        self._runtime = runtime_ctx

    def _refresh_live_datapoint(self, *, image_rgb: np.ndarray | None = None) -> Any:
        isaac_env = self._runtime.isaac_env
        isaac_env.update_metrics()
        return isaac_env.getObservedDatapoint(self._runtime.sim_env_config, RESET_DATAPOINT=False, image_rgb=image_rgb)

    def get_current_scene_graph(
        self,
        *,
        state_name: str | None = None,
        manual_scene_graph: dict[str, Any] | None = None,
        image_rgb: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], Any | None]:
        if manual_scene_graph is not None:
            return manual_scene_graph, None
        datapoint = self._refresh_live_datapoint(image_rgb=image_rgb)
        scene_graph = self._runtime.scene_graph_translator.datapoint_to_scene_graph_json(
            self._runtime.sim_env_config,
            datapoint,
            state_name=state_name,
        )
        return scene_graph, datapoint


class LazyTALPlanner:
    def __init__(self, runtime_ctx: TALRuntimeContext):
        self._runtime = runtime_ctx
        self._plan_lock = threading.Lock()

    def plan_first_action(
        self,
        user_instruction: str,
        current_scene_graph_json: Mapping[str, Any],
        start_node: Any | None = None,
    ) -> TALPlanResult:
        with self._plan_lock:
            planner_config = self._runtime.planner_env_config
            current_state_graph = self._runtime.scene_graph_json_to_dgl(planner_config, dict(current_scene_graph_json))
            current_state_graph = current_state_graph.to(planner_config.device)

            world_num = 0
            graph_world_name = getattr(planner_config, "graph_world_name", "")
            digits = "".join(ch for ch in str(graph_world_name) if ch.isdigit())
            if digits:
                world_num = int(digits)

            result = self._runtime.plan_with_natural_language_instruction(
                planner_config,
                model_action=self._runtime.model_action,
                model_extract_feature=self._runtime.model_action_effect,
                action_effect_features=self._runtime.action_effect_features,
                instruction=user_instruction,
                world_num=world_num,
                start_node=start_node,
                current_state_graph=current_state_graph,
                current_scene_graph_json=dict(current_scene_graph_json),
                qwen_model_name=self._runtime.qwen_model,
                qwen_api_key=os.getenv(self._runtime.qwen_api_key_env) if self._runtime.qwen_api_key_env else None,
                candidate_action_num=self._runtime.candidate_action_num,
                select_from_candidate=self._runtime.select_from_candidate,
                trajectory_length=self._runtime.max_planning_steps,
                with_pca=True,
            )

        predicted_actions = list(result.get("predicted_actions", []))
        first_action = format_tal_action(predicted_actions[0]) if predicted_actions else None
        return TALPlanResult(
            status=result.get("status", "Unknown"),
            first_action_text=first_action,
            predicted_actions=predicted_actions,
            current_scene_graph_json=result.get("current_scene_graph_json"),
            goal_scene_graph_json=result.get("goal_scene_graph_json"),
            error=result.get("error"),
        )


def load_manual_scene_graph(path_str: str) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _try_read_camera_rgb(camera: Any) -> np.ndarray | None:
    image = camera.get_rgba()
    image = np.asarray(image) if image is not None else None
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] < 3:
        return None
    rgb = image[:, :, :3]
    if rgb.dtype == np.float32:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8, copy=False)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_camera_rgb(camera: Any, camera_name: str) -> np.ndarray:
    retries = max(int(os.environ.get("TAL_ONLINE_CAMERA_RETRIES", "6")), 1)
    for _ in range(retries):
        image_bgr = _try_read_camera_rgb(camera)
        if image_bgr is not None:
            return image_bgr
        time.sleep(0.02)
    raise RuntimeError(f"Camera {camera_name} returned no valid RGBA frame after {retries} retries")


def _parse_nav_map_yaml(yaml_path: Path) -> tuple[float, float, float, str]:
    import yaml

    data = yaml.safe_load(yaml_path.read_text())
    return float(data["resolution"]), float(data["origin"][0]), float(data["origin"][1]), str(data["image"])


def load_nav_occupancy_map(yaml_path: Path) -> NavOccupancyMap:
    resolution, origin_x, origin_y, image_name = _parse_nav_map_yaml(yaml_path)
    image = cv2.imread(str(yaml_path.parent / image_name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load navigation map image for {yaml_path}")
    return NavOccupancyMap(resolution=resolution, origin_x=origin_x, origin_y=origin_y, image=image)


def project_nav_goal_to_free_space(occupancy_map: NavOccupancyMap, x: float, y: float, *, clearance_m: float) -> tuple[float, float]:
    if occupancy_map.has_clearance(x, y, clearance_m):
        return x, y
    max_radius = 40
    for radius in range(1, max_radius + 1):
        steps = max(16, radius * 8)
        for k in range(steps):
            theta = 2.0 * math.pi * (k / steps)
            x2 = x + math.cos(theta) * radius * occupancy_map.resolution
            y2 = y + math.sin(theta) * radius * occupancy_map.resolution
            if occupancy_map.has_clearance(x2, y2, clearance_m):
                return x2, y2
    return x, y


def resolve_tal_object_name(runtime_ctx: TALRuntimeContext, object_name: str) -> str:
    name = str(object_name).strip()
    env_cfg = runtime_ctx.sim_env_config
    if name in getattr(env_cfg, "all_objects", []):
        return name
    lowered = name.lower()
    for candidate in getattr(env_cfg, "all_objects", []):
        if candidate.lower() == lowered:
            return candidate
    return name


def infer_navigation_approach_distance(runtime_ctx: TALRuntimeContext, object_name: str, *, source_action_name: str | None = None) -> float:
    env_cfg = runtime_ctx.sim_env_config
    overrides = getattr(env_cfg, "nav_approach_distance_overrides", {})
    if object_name in overrides:
        return float(overrides[object_name])
    source = (source_action_name or "").lower()
    if source == "pick":
        return float(getattr(env_cfg, "pick_approach_distance", getattr(env_cfg, "base_approach_distance", 0.50)))
    if source == "pushto":
        return float(getattr(env_cfg, "push_approach_distance", getattr(env_cfg, "base_approach_distance", 0.50)))
    if source == "picknplaceaonb":
        return float(getattr(env_cfg, "pick_approach_distance", getattr(env_cfg, "base_approach_distance", 0.50)))
    return float(getattr(env_cfg, "base_approach_distance", 0.50))


def infer_navigation_approach_direction(runtime_ctx: TALRuntimeContext, object_name: str, robot_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    overrides = getattr(runtime_ctx.sim_env_config, "nav_approach_direction_overrides", {})
    if object_name in overrides:
        direction = np.asarray(overrides[object_name], dtype=np.float32)
    else:
        direction = robot_xy.astype(np.float32) - target_xy.astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return direction / norm


def build_navigation_goal(runtime_ctx: TALRuntimeContext, object_name: str, *, source_action_name: str | None = None) -> NavigationGoal:
    isaac_env = runtime_ctx.isaac_env
    isaac_env.update_metrics()
    metrics = getattr(isaac_env, "metrics", None)
    if not isinstance(metrics, Mapping):
        raise RuntimeError("TAL isaac_env metrics is unavailable after update_metrics()")
    resolved_name = resolve_tal_object_name(runtime_ctx, object_name)
    target_pos = np.asarray(metrics[resolved_name][0], dtype=np.float32)
    robot_pos = np.asarray(metrics["husky"][0], dtype=np.float32)
    target_xy = target_pos[:2]
    robot_xy = robot_pos[:2]
    stop_distance = infer_navigation_approach_distance(runtime_ctx, resolved_name, source_action_name=source_action_name)
    direction = infer_navigation_approach_direction(runtime_ctx, resolved_name, robot_xy, target_xy)
    goal_xy = target_xy + direction * stop_distance
    occupancy_map = load_nav_occupancy_map(NAV_MAP_YAML_PATH)
    footprint = getattr(runtime_ctx.sim_env_config, "lidar_footprint_overrides", {}).get(resolved_name, [0.24, 0.24])
    clearance_m = max(float(footprint[0]), float(footprint[1]), 0.24) * 0.5 + 0.06
    goal_x, goal_y = project_nav_goal_to_free_space(occupancy_map, float(goal_xy[0]), float(goal_xy[1]), clearance_m=clearance_m)
    yaw = math.atan2(float(target_xy[1] - goal_y), float(target_xy[0] - goal_x))
    return NavigationGoal(x=float(goal_x), y=float(goal_y), yaw=float(yaw))


class RobotRootPoseController:
    def __init__(self, prim: Any):
        self._prim = prim

    def get_world_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if hasattr(self._prim, "get_world_poses"):
            positions, orientations = self._prim.get_world_poses()
            if positions is None or len(positions) == 0:
                return None, None
            position = np.asarray(positions[0], dtype=np.float32)
            orientation = None
            if orientations is not None and len(orientations) > 0:
                orientation = np.asarray(orientations[0], dtype=np.float32)
            return position, orientation
        elif hasattr(self._prim, "get_world_pose"):
            position, orientation = self._prim.get_world_pose()
            return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)
        return None, None

    def set_world_pose(
        self,
        *,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> None:
        current_position, current_orientation = self.get_world_pose()
        if current_position is None:
            raise RuntimeError("Failed to read root prim world pose.")
        target_position = current_position if position is None else np.asarray(position, dtype=np.float32)
        target_orientation = current_orientation if orientation is None else np.asarray(orientation, dtype=np.float32)
        if hasattr(self._prim, "set_world_poses"):
            positions = target_position.reshape(1, 3)
            orientations = None if target_orientation is None else target_orientation.reshape(1, 4)
            self._prim.set_world_poses(positions=positions, orientations=orientations)
        elif hasattr(self._prim, "set_world_pose"):
            self._prim.set_world_pose(position=target_position, orientation=target_orientation)

    def set_linear_velocity(self, velocity: np.ndarray) -> None:
        if hasattr(self._prim, "set_linear_velocities"):
            try:
                self._prim.set_linear_velocities(np.asarray(velocity, dtype=np.float32).reshape(1, 3))
            except Exception:
                pass

    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        if hasattr(self._prim, "set_angular_velocities"):
            try:
                self._prim.set_angular_velocities(np.asarray(velocity, dtype=np.float32).reshape(1, 3))
            except Exception:
                pass


class RobotRootPoseGuard:
    def __init__(self, robot: Any, robot_root_controller: RobotRootPoseController):
        self._robot = robot
        self._robot_root_controller = robot_root_controller
        self._enabled = os.environ.get("TAL_NAV_ENFORCE_ROOT_POSE", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        position, orientation = self._robot_root_controller.get_world_pose()
        self._position = None if position is None else np.asarray(position, dtype=np.float32)
        self._orientation = None if orientation is None else np.asarray(orientation, dtype=np.float32)

    def _zero_root_velocity(self) -> None:
        zero = np.zeros(3, dtype=np.float32)
        self._robot_root_controller.set_linear_velocity(zero)
        self._robot_root_controller.set_angular_velocity(zero)
        if hasattr(self._robot, "set_linear_velocity"):
            try:
                self._robot.set_linear_velocity(zero)
                self._robot.set_angular_velocity(zero)
            except Exception:
                pass

    def enforce_after_step(self) -> None:
        if not self._enabled or self._position is None:
            return
        self._zero_root_velocity()
        self._robot_root_controller.set_world_pose(position=self._position, orientation=self._orientation)
        self._zero_root_velocity()


class IsaacNavBridge:
    def __init__(self, robot: Any, robot_root_controller: RobotRootPoseController):
        self._robot = robot
        self._robot_root_controller = robot_root_controller
        self._cmd_vx = 0.0
        self._cmd_vw = 0.0
        self._applied_vx = 0.0
        self._applied_vw = 0.0
        self._sim_time_s = 0.0
        self._active_goal: NavigationGoal | None = None
        self._root_yaw_offset = float(os.environ.get("TAL_ROBOT_ROOT_YAW_OFFSET", "0.0"))
        self._max_linear_accel = max(float(os.environ.get("TAL_NAV_MAX_LINEAR_ACCEL", "0.20")), 1e-4)
        self._max_angular_accel = max(float(os.environ.get("TAL_NAV_MAX_ANGULAR_ACCEL", "0.50")), 1e-4)
        self._max_translation_step_m = max(float(os.environ.get("TAL_NAV_MAX_TRANSLATION_STEP_M", "0.01")), 1e-4)
        self._max_yaw_step_rad = max(float(os.environ.get("TAL_NAV_MAX_YAW_STEP_RAD", "0.03")), 1e-4)
        self._root_update_interval_s = max(float(os.environ.get("TAL_NAV_ROOT_UPDATE_INTERVAL_S", "0.10")), 1e-3)
        self._root_update_accum_s = 0.0
        self._enforce_root_pose = os.environ.get("TAL_NAV_ENFORCE_ROOT_POSE", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_socket.bind(("127.0.0.1", 0))
        self._cmd_socket.setblocking(False)
        self._idle_root_hold_enabled = False
        self._cmd_port = int(self._cmd_socket.getsockname()[1])
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        self._state_port = int(probe.getsockname()[1])
        probe.close()
        self._bridge_process: subprocess.Popen[str] | None = None
        initial_position, _ = self._robot_root_controller.get_world_pose()
        self._base_z = float(initial_position[2]) if initial_position is not None else 0.0
        position, orientation = self._robot_root_controller.get_world_pose()
        if position is None:
            raise RuntimeError("Failed to read initial robot world pose for navigation bridge.")
        initial_root_roll, initial_root_pitch, initial_root_yaw = quaternion_wxyz_to_rpy(orientation)
        flatten_root_attitude = os.environ.get("TAL_NAV_FLATTEN_ROOT_ATTITUDE", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._flatten_root_attitude = flatten_root_attitude
        if flatten_root_attitude:
            initial_root_roll = 0.0
            initial_root_pitch = 0.0
        if abs(self._root_yaw_offset) > 1e-6:
            corrected_root_yaw = normalize_angle(initial_root_yaw + self._root_yaw_offset)
            self._robot_root_controller.set_world_pose(
                position=np.asarray(position, dtype=np.float32),
                orientation=rpy_to_quaternion_wxyz(initial_root_roll, initial_root_pitch, corrected_root_yaw),
            )
            self._zero_root_velocity()
            position, orientation = self._robot_root_controller.get_world_pose()
        self._xform_x = float(position[0])
        self._xform_y = float(position[1])
        self._xform_z = float(position[2])
        self._root_roll, self._root_pitch, root_yaw = quaternion_wxyz_to_rpy(orientation)
        if flatten_root_attitude:
            self._root_roll = 0.0
            self._root_pitch = 0.0
        self._xform_yaw = normalize_angle(root_yaw - self._root_yaw_offset)
        self._x = self._xform_x
        self._y = self._xform_y
        self._z = self._xform_z
        self._yaw = self._xform_yaw
        self._apply_root_pose()
        self._start_bridge_process()

    def _zero_root_velocity(self) -> None:
        zero = np.zeros(3, dtype=np.float32)
        self._robot_root_controller.set_linear_velocity(zero)
        self._robot_root_controller.set_angular_velocity(zero)
        if hasattr(self._robot, "set_linear_velocity"):
            try:
                self._robot.set_linear_velocity(zero)
                self._robot.set_angular_velocity(zero)
            except Exception:
                pass

    def _root_orientation(self) -> np.ndarray:
        return rpy_to_quaternion_wxyz(
            self._root_roll,
            self._root_pitch,
            normalize_angle(self._xform_yaw + self._root_yaw_offset),
        )

    def sync_from_current_root(self) -> None:
        position, orientation = self._robot_root_controller.get_world_pose()
        if position is None:
            return
        position = np.asarray(position, dtype=np.float32)
        roll, pitch, yaw = quaternion_wxyz_to_rpy(orientation)
        if self._flatten_root_attitude:
            roll = 0.0
            pitch = 0.0
        self._base_z = float(position[2])
        self._xform_x = float(position[0])
        self._xform_y = float(position[1])
        self._xform_z = self._base_z
        self._root_roll = roll
        self._root_pitch = pitch
        self._xform_yaw = normalize_angle(yaw - self._root_yaw_offset)
        self._x = self._xform_x
        self._y = self._xform_y
        self._z = self._base_z
        self._yaw = self._xform_yaw
        self._root_update_accum_s = 0.0
        self._idle_root_hold_enabled = True
        self._apply_root_pose()

    def _apply_root_pose(self) -> None:
        self._zero_root_velocity()
        self._robot_root_controller.set_world_pose(
            position=np.array([self._xform_x, self._xform_y, self._xform_z], dtype=np.float32),
            orientation=self._root_orientation(),
        )
        self._zero_root_velocity()

    def enforce_root_pose_after_step(self) -> None:
        if self._enforce_root_pose and (self._active_goal is not None or self._idle_root_hold_enabled):
            self._apply_root_pose()

    def enforce_after_step(self) -> None:
        self.enforce_root_pose_after_step()

    @staticmethod
    def _step_towards(current: float, target: float, max_delta: float) -> float:
        if target > current:
            return min(current + max_delta, target)
        return max(current - max_delta, target)

    def _start_bridge_process(self) -> None:
        self._bridge_stdout_path = Path(f"/tmp/isaac_nav_bridge_{self._state_port}.out.log")
        self._bridge_stderr_path = Path(f"/tmp/isaac_nav_bridge_{self._state_port}.err.log")
        cmd = [
            "bash",
            "-lc",
            (
                "mkdir -p /tmp/ros_log_isaac_nav_bridge && "
                "export ROS_LOG_DIR=/tmp/ros_log_isaac_nav_bridge && "
                "unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH && "
                "source /opt/ros/jazzy/setup.bash && "
                f"/usr/bin/python3 /root/gpufree-data/code/tal-vla/isaac_nav_bridge_runner.py --host 127.0.0.1 --state-port {self._state_port} --cmd-port {self._cmd_port}"
            ),
        ]
        stdout_file = self._bridge_stdout_path.open("w", encoding="utf-8")
        stderr_file = self._bridge_stderr_path.open("w", encoding="utf-8")
        self._bridge_process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True)
        time.sleep(2.5)
        if self._bridge_process.poll() is not None:
            stdout_file.close()
            stderr_file.close()
            stdout_text = self._bridge_stdout_path.read_text(encoding="utf-8", errors="ignore") if self._bridge_stdout_path.exists() else ""
            stderr_text = self._bridge_stderr_path.read_text(encoding="utf-8", errors="ignore") if self._bridge_stderr_path.exists() else ""
            raise RuntimeError(
                "Isaac navigation bridge ROS subprocess exited immediately; "
                f"stdout={stdout_text!r} stderr={stderr_text!r}"
            )

    def close(self) -> None:
        try:
            if self._bridge_process is not None and self._bridge_process.poll() is None:
                self._bridge_process.terminate()
                try:
                    self._bridge_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._bridge_process.kill()
        except Exception:
            pass
        try:
            self._cmd_socket.close()
        except OSError:
            pass

    def set_active_goal(self, goal: NavigationGoal | None) -> None:
        self._active_goal = goal

    def _poll_cmd(self) -> None:
        while True:
            try:
                packet, _ = self._cmd_socket.recvfrom(65535)
            except BlockingIOError:
                break
            payload = json.loads(packet.decode("utf-8"))
            self._cmd_vx = float(payload.get("vx", 0.0))
            self._cmd_vw = float(payload.get("vw", 0.0))

    def _publish_state(self) -> None:
        payload = {
            "sim_time_s": self._sim_time_s,
            "x": float(self._x),
            "y": float(self._y),
            "z": float(self._z),
            "yaw": float(self._yaw),
            "vx": float(self._applied_vx),
            "vw": float(self._applied_vw),
            "scan_angle_min": -math.pi / 2.0,
            "scan_angle_max": math.pi / 2.0,
            "scan_angle_increment": math.pi / 180.0,
            "scan_range_min": 0.12,
            "scan_range_max": 3.5,
            "scan_ranges": [3.0] * 181,
        }
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), ("127.0.0.1", self._state_port))
        finally:
            sock.close()

    def advance(self, dt: float) -> None:
        if self._active_goal is None:
            self._cmd_vx = 0.0
            self._cmd_vw = 0.0
            self._applied_vx = 0.0
            self._applied_vw = 0.0
            self._sim_time_s += dt
            self._root_update_accum_s = 0.0
            if self._idle_root_hold_enabled:
                self._apply_root_pose()
            self._publish_state()
            return
        self._poll_cmd()
        self._sim_time_s += dt
        self._root_update_accum_s += dt
        if self._root_update_accum_s < self._root_update_interval_s:
            self._publish_state()
            return
        effective_dt = self._root_update_accum_s
        self._root_update_accum_s = 0.0
        target_vx = float(self._cmd_vx)
        target_vw = float(self._cmd_vw)
        self._applied_vx = self._step_towards(self._applied_vx, target_vx, self._max_linear_accel * effective_dt)
        self._applied_vw = self._step_towards(self._applied_vw, target_vw, self._max_angular_accel * effective_dt)
        yaw_step = float(np.clip(self._applied_vw * effective_dt, -self._max_yaw_step_rad, self._max_yaw_step_rad))
        self._yaw = normalize_angle(self._yaw + yaw_step)
        translation_step = float(
            np.clip(self._applied_vx * effective_dt, -self._max_translation_step_m, self._max_translation_step_m)
        )
        self._x += math.cos(self._yaw) * translation_step
        self._y += math.sin(self._yaw) * translation_step
        self._z = self._base_z
        self._xform_x = self._x
        self._xform_y = self._y
        self._xform_z = self._base_z
        self._xform_yaw = self._yaw
        self._apply_root_pose()
        self._publish_state()

    def settle_to_goal_pose(self, goal: NavigationGoal) -> None:
        self._cmd_vx = 0.0
        self._cmd_vw = 0.0
        self._applied_vx = 0.0
        self._applied_vw = 0.0
        self._x = float(goal.x)
        self._y = float(goal.y)
        self._z = self._base_z
        self._yaw = normalize_angle(goal.yaw)
        self._xform_x = self._x
        self._xform_y = self._y
        self._xform_z = self._base_z
        self._xform_yaw = self._yaw
        self._idle_root_hold_enabled = True
        self._apply_root_pose()


class SubprocessNav2GoalClient:
    def __init__(self, local_setup_path: str):
        self._local_setup_path = local_setup_path

    def send_goal(self, goal: NavigationGoal, *, result_timeout: float, server_timeout: float) -> PendingNavigation:
        request = PendingNavigation(goal=goal)
        cmd = [
            "bash",
            "-lc",
            (
                "mkdir -p /tmp/ros_log_nav2_goal && "
                "export ROS_LOG_DIR=/tmp/ros_log_nav2_goal && "
                "unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH && "
                "source /opt/ros/jazzy/setup.bash && "
                f"source {shlex.quote(self._local_setup_path)} && "
                f"/usr/bin/python3 {shlex.quote('/root/gpufree-data/code/tal-vla/nav2_goal_runner.py')} "
                f"--x {goal.x} --y {goal.y} --yaw {goal.yaw} --frame-id {goal.frame_id} "
                f"--server-timeout {server_timeout} --result-timeout {result_timeout}"
            ),
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def _wait() -> None:
            stdout, stderr = process.communicate()
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            payload = None
            if lines:
                try:
                    payload = json.loads(lines[-1])
                except Exception:
                    payload = None
            request.accepted = True
            if payload is None:
                request.success = False
                request.error = stderr.strip() or stdout.strip() or f"Nav2 goal runner exited with code {process.returncode}"
                request.status = process.returncode
            else:
                request.success = bool(payload.get("success", False))
                request.status = int(payload.get("status", process.returncode or 0))
                request.error = payload.get("error")
            request.done_event.set()

        threading.Thread(target=_wait, daemon=True).start()
        return request

    def cancel(self, request: PendingNavigation) -> None:
        request.error = request.error or "Nav2 goal cancelled"
        request.done_event.set()


def advance_simulation(world: Any, nav_bridge: IsaacNavBridge | None, dt: float, *, render: bool) -> None:
    if nav_bridge is not None:
        nav_bridge.advance(dt)
    _step_world_with_root_guard(world, render=render, root_pose_guard=nav_bridge)


def warm_up_cameras(
    world: Any,
    nav_bridge: IsaacNavBridge | None,
    cameras: dict[str, Any],
    dt: float,
    *,
    root_pose_guard: Any | None = None,
) -> None:
    for _ in range(30):
        if nav_bridge is None and root_pose_guard is not None:
            _step_world_with_root_guard(world, render=True, root_pose_guard=root_pose_guard)
        else:
            advance_simulation(world, nav_bridge, dt, render=True)
        for camera in cameras.values():
            try:
                camera.get_rgba()
            except Exception:
                pass


def wait_for_camera_frames(
    cam_high: Any,
    cam_wrist: Any,
    world: Any,
    *,
    headless: bool,
    cam_tal: Any | None = None,
    root_pose_guard: Any | None = None,
) -> dict[str, np.ndarray]:
    preflight_retries = max(int(os.environ.get("TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES", "120")), 1)
    for _ in range(preflight_retries):
        _step_world_with_root_guard(world, render=not headless, root_pose_guard=root_pose_guard)
        high = _try_read_camera_rgb(cam_high)
        wrist = _try_read_camera_rgb(cam_wrist)
        tal = _try_read_camera_rgb(cam_tal) if cam_tal is not None else high
        if high is not None and wrist is not None and tal is not None:
            return {"cam_high": high, "cam_wrist": wrist, "cam_tal": tal}
        time.sleep(0.02)
    raise RuntimeError(
        f"Camera preflight failed after {preflight_retries} retries; required camera frames are not all ready"
    )


def capture_rgb_images(cam_high: Any, cam_wrist: Any, cam_tal: Any | None = None) -> dict[str, np.ndarray]:
    images = {
        "cam_high": _read_camera_rgb(cam_high, "cam_high"),
        "cam_wrist": _read_camera_rgb(cam_wrist, "cam_wrist"),
    }
    images["cam_tal"] = _read_camera_rgb(cam_tal, "cam_tal") if cam_tal is not None else images["cam_high"]
    return images


def read_robot_state(robot: Any, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_joint_pos = robot.get_joint_positions()
    all_dof_names = robot.dof_names
    ordered_state = []
    for name in joint_names:
        if name not in all_dof_names:
            raise ValueError(f"Joint {name} not found in simulation DOF names: {all_dof_names}")
        idx = all_dof_names.index(name)
        ordered_state.append(all_joint_pos[idx])
    return np.array(ordered_state, dtype=np.float32), all_joint_pos, all_dof_names


def should_replan(step_idx: int, replan_every_n_steps: int) -> bool:
    if replan_every_n_steps <= 1:
        return True
    return step_idx % replan_every_n_steps == 0


def infer_action(policy_client: Any, images: dict[str, np.ndarray], state: np.ndarray, fused_prompt: str) -> np.ndarray:
    obs = {
        "observation/images/cam_high": images["cam_high"],
        "observation/images/cam_wrist": images["cam_wrist"],
        "observation/state": state,
        "prompt": fused_prompt,
    }
    result = policy_client.infer(obs)
    return result["actions"][0]


def apply_robot_action(
    robot: Any,
    world: Any,
    target_action: np.ndarray,
    target_indices: np.ndarray,
    ArticulationAction: Any,
    *,
    nav_bridge: IsaacNavBridge | None = None,
    dt: float = 0.05,
    render: bool = False,
) -> None:
    target_action = np.asarray(target_action, dtype=np.float32)
    target_indices = np.asarray(target_indices, dtype=np.int32)

    action_cmd = ArticulationAction(
        joint_positions=target_action,
        joint_indices=target_indices,
    )
    use_position_targets = hasattr(robot, "set_joint_position_targets")
    use_direct_set = os.environ.get("TAL_ONLINE_DIRECT_SET_JOINTS", "0").lower() in {"1", "true", "yes", "on"}
    can_direct_set = hasattr(robot, "set_joint_positions")

    control_substeps = max(int(os.environ.get("TAL_ONLINE_CONTROL_SUBSTEPS", "4")), 1)
    for _ in range(control_substeps):
        if use_direct_set and can_direct_set:
            robot.set_joint_positions(target_action, joint_indices=target_indices)
        elif use_position_targets:
            robot.set_joint_position_targets(target_action, joint_indices=target_indices)
        else:
            robot.apply_action(action_cmd)
        advance_simulation(world, nav_bridge, dt, render=render)

    if os.environ.get("TAL_ONLINE_DEBUG_CONTROL", "0").lower() in {"1", "true", "yes", "on"}:
        post_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
        print(f"[ControlDebug] target={target_action.tolist()} post_state={post_state.tolist()}")


def smoothly_move_robot_root(
    robot: Any,
    world: Any,
    target_position: np.ndarray,
    *,
    num_steps: int = 240,
) -> None:
    current_position, current_orientation = robot.get_world_pose()
    if current_position is None:
        raise RuntimeError("Failed to read world pose for robot")

    start_position = np.asarray(current_position, dtype=np.float32)
    start_orientation = None
    if current_orientation is not None:
        start_orientation = np.asarray(current_orientation, dtype=np.float32)

    target_position = np.asarray(target_position, dtype=np.float32)
    print(f"[InitMove] Original Target Z: {target_position[2]:.4f}, Overriding to Safe Z: {start_position[2]:.4f}")
    target_position[2] = start_position[2]

    print(f"[InitMove] Start world position: {start_position.tolist()}")
    print(f"[InitMove] Target world position: {target_position.tolist()}")

    for step in range(num_steps):
        alpha = float(step + 1) / float(num_steps)
        smooth_alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
        interpolated_position = start_position + smooth_alpha * (target_position - start_position)
        lift_height = np.sin(alpha * np.pi) * 0.03
        interpolated_position[2] += lift_height
        if start_orientation is None:
            robot.set_world_pose(position=interpolated_position)
        else:
            robot.set_world_pose(
                position=interpolated_position,
                orientation=start_orientation,
            )
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        world.step(render=True)

    for _ in range(60):
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        advance_simulation(world, None, dt=0.05, render=False)

    final_position, _ = robot.get_world_pose()
    if final_position is None:
        raise RuntimeError("Failed to verify final world pose for robot")
    print(f"[InitMove] Final world position: {np.asarray(final_position, dtype=np.float32).tolist()}")


def warm_up_robot(
    robot: Any,
    world: Any,
    target_indices: np.ndarray,
    ArticulationAction: Any,
    *,
    nav_bridge: IsaacNavBridge | None = None,
    dt: float = 0.05,
    render: bool = False,
    root_pose_guard: Any | None = None,
) -> None:
    target_state = _get_initial_robot_state()
    start_positions = robot.get_joint_positions()[target_indices]
    num_steps = 240
    warmup_direct_set = os.environ.get("TAL_ONLINE_WARMUP_DIRECT_SET_JOINTS", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    joint_indices_i32 = target_indices.astype(np.int32)
    for i in range(num_steps):
        alpha = (i + 1) / float(num_steps)
        interpolated_positions = start_positions + alpha * (target_state - start_positions)
        if warmup_direct_set and hasattr(robot, "set_joint_positions"):
            robot.set_joint_positions(interpolated_positions, joint_indices=joint_indices_i32)
        else:
            step_action = ArticulationAction(
                joint_positions=interpolated_positions,
                joint_indices=joint_indices_i32,
            )
            robot.apply_action(step_action)
        if nav_bridge is None and root_pose_guard is not None:
            _step_world_with_root_guard(world, render=render, root_pose_guard=root_pose_guard)
        else:
            advance_simulation(world, nav_bridge, dt, render=render)

    final_action = ArticulationAction(
        joint_positions=target_state,
        joint_indices=joint_indices_i32,
    )
    for _ in range(60):
        if warmup_direct_set and hasattr(robot, "set_joint_positions"):
            robot.set_joint_positions(target_state, joint_indices=joint_indices_i32)
        else:
            robot.apply_action(final_action)
        if nav_bridge is None and root_pose_guard is not None:
            _step_world_with_root_guard(world, render=False, root_pose_guard=root_pose_guard)
        else:
            advance_simulation(world, nav_bridge, dt, render=False)


def main() -> None:
    print(f"--> Current Prompt: {args.prompt}")
    runtime_ctx = initialize_tal_runtime(
        TALControllerConfig(
            tal_root=args.tal_root,
            qwen_model=args.qwen_model,
            qwen_api_key_env=args.qwen_api_key_env,
            headless=args.headless,
        )
    )
    simulation_app = runtime_ctx.simulation_app

    try:
        from isaacsim.core.api import World
    except ModuleNotFoundError:
        from omni.isaac.core import World

    try:
        from isaacsim.core.prims import SingleArticulation as Articulation
    except ModuleNotFoundError:
        try:
            from omni.isaac.core.articulations import Articulation
        except ModuleNotFoundError:
            from isaacsim.core.experimental.prims import Articulation

    try:
        from isaacsim.core.prims import XFormPrim
    except ModuleNotFoundError:
        try:
            from omni.isaac.core.prims import XFormPrim
        except ModuleNotFoundError:
            from isaacsim.core.experimental.prims import XformPrim as XFormPrim

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ModuleNotFoundError:
        from omni.isaac.core.utils.types import ArticulationAction

    try:
        from isaacsim.sensors.camera import Camera
    except ModuleNotFoundError:
        from omni.isaac.sensor import Camera

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    robot_usd_name = runtime_ctx.sim_env_config.tal_to_usd["husky"]
    robot_prim_path = f"/World/{robot_usd_name}"
    print(f"Loaded TAL scene: {runtime_ctx.sim_env_config.scene_usd_path}")
    print(f"Robot prim resolved from TAL config: {robot_prim_path}")

    try:
        world = World(stage_units_in_meters=1.0)
    except TypeError:
        world = World()
    robot = world.scene.add(Articulation(prim_path=robot_prim_path, name="firefighter"))

    cam_high = Camera(prim_path=CAMERA_HIGH_PATH, resolution=(224, 224))
    cam_wrist = Camera(prim_path=CAMERA_WRIST_PATH, resolution=(224, 224))
    cam_tal = None
    if CAMERA_TAL_PATH != CAMERA_HIGH_PATH:
        cam_tal = Camera(prim_path=CAMERA_TAL_PATH, resolution=(224, 224))
    cam_high.initialize()
    cam_wrist.initialize()
    if cam_tal is not None:
        cam_tal.initialize()

    tal_planner = LazyTALPlanner(runtime_ctx)
    scene_graph_provider = TALSceneGraphProvider(runtime_ctx)
    manual_scene_graph = load_manual_scene_graph(args.manual_scene_graph_json)

    saved_pose_state, saved_pose_indices = _capture_current_robot_pose(robot, robot_prim_path)
    # 🌟 核心修改 1：直接通过真实的物理关节体 (Articulation) 作为控制器的底层输入，而不是未初始化的原始 XFormPrim！
    # 这能从根本上避免 PhysX 物理计算状态与 USD 节点的冲突，彻底斩断小车向前的“闪现和翘头漂移”现象。
    robot_root_controller = RobotRootPoseController(robot)
    early_root_pose_guard = RobotRootPoseGuard(robot, robot_root_controller)

    if _should_reset_world():
        world.reset()
        early_root_pose_guard.enforce_after_step()
    _initialize_robot_handles_if_needed(robot, world, headless=args.headless, root_pose_guard=early_root_pose_guard)
    if _should_reinitialize_robot():
        _reinitialize_robot_if_possible(robot)
    _restore_robot_pose_if_available(
        robot,
        world,
        saved_pose_state,
        saved_pose_indices,
        ArticulationAction,
        headless=args.headless,
        root_pose_guard=early_root_pose_guard,
    )

    sim_dof_names = robot.dof_names
    target_indices = []
    for name in JOINT_NAMES_IN_ORDER:
        if name in sim_dof_names:
            target_indices.append(sim_dof_names.index(name))
        else:
            print(f"Warning: joint {name} was not found in simulation.")
    target_indices = np.array(target_indices, dtype=np.int32)

    nav_bridge = IsaacNavBridge(robot, robot_root_controller)

    if _should_move_robot_root():
        smoothly_move_robot_root(robot, world, ROBOT_START_WORLD_POSITION)
        if _should_reinitialize_robot():
            _reinitialize_robot_if_possible(robot)
    if _should_warmup_robot():
        warm_up_robot(
            robot,
            world,
            target_indices,
            ArticulationAction,
            nav_bridge=None,
            dt=args.nav_control_dt,
            render=not args.headless,
            root_pose_guard=early_root_pose_guard,
        )
        if _should_reinitialize_robot():
            _reinitialize_robot_if_possible(robot)

    policy = None
    nav_client = SubprocessNav2GoalClient("/root/gpufree-data/code/tal-vla/robot_ws/install/local_setup.bash")
    warmup_steps = max(int(args.nav_warmup_sec / max(args.nav_control_dt, 1e-3)), 1)
    for _ in range(warmup_steps):
        _step_world_with_root_guard(world, render=not args.headless, root_pose_guard=early_root_pose_guard)

    print("Starting TAL(native scene graph) + OpenPI closed-loop inference...")

    latest_subtask = None
    latest_fused_prompt = args.prompt
    latest_parsed_subtask: ParsedTALSubtask | None = None
    latest_raw_subtask: str | None = None
    latest_raw_parsed_subtask: ParsedTALSubtask | None = None
    completed_navigation_subtasks: set[tuple[str, tuple[str, ...]]] = set()
    skip_replan_once = False
    force_replan = True
    latest_images = wait_for_camera_frames(
        cam_high,
        cam_wrist,
        world,
        headless=args.headless,
        cam_tal=cam_tal,
        root_pose_guard=early_root_pose_guard,
    )
    warm_up_cameras(
        world,
        None,
        {"cam_high": cam_high, "cam_wrist": cam_wrist, **({"cam_tal": cam_tal} if cam_tal is not None else {})},
        args.nav_control_dt,
        root_pose_guard=early_root_pose_guard,
    )
    step_idx = 0

    try:
        while True:
            print(f"[Loop] entering step {step_idx}")
            advance_simulation(world, nav_bridge, args.nav_control_dt, render=not args.headless)
            if args.max_steps >= 0 and step_idx >= args.max_steps:
                print("Reached max steps, exiting.")
                break

            print(f"[Step {step_idx}] Capturing RGB images...")
            try:
                images = capture_rgb_images(cam_high, cam_wrist, cam_tal=cam_tal)
                latest_images = images
            except RuntimeError as exc:
                if latest_images is None:
                    raise
                print(f"[CameraWarning] {exc}; reuse previous RGB frame", flush=True)
                images = latest_images
            print(f"[Step {step_idx}] Reading robot state...")
            current_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
            print(f"[Step {step_idx}] Robot state: {current_state.tolist()}")

            if skip_replan_once:
                skip_replan_once = False
            elif force_replan or should_replan(step_idx, args.replan_every_n_steps):
                scene_graph_state_name = args.tal_world_state_name if step_idx == 0 else None
                print(
                    f"[Step {step_idx}] Replanning triggered. "
                    f"scene_graph_state_name={scene_graph_state_name!r}, "
                    f"manual_scene_graph={'yes' if manual_scene_graph is not None else 'no'}"
                )
                if os.environ.get("TAL_ONLINE_DEBUG_REPLAN", "0").lower() in {"1", "true", "yes", "on"}:
                    pre_replan_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
                    print(f"[ReplanDebug] before_scene_graph state={pre_replan_state.tolist()}")
                print(f"[Step {step_idx}] Building current scene graph from TAL runtime...")
                current_scene_graph, current_datapoint = scene_graph_provider.get_current_scene_graph(
                    state_name=scene_graph_state_name,
                    manual_scene_graph=manual_scene_graph,
                    image_rgb=cv2.cvtColor(images["cam_tal"], cv2.COLOR_BGR2RGB),
                )
                if os.environ.get("TAL_ONLINE_DEBUG_REPLAN", "0").lower() in {"1", "true", "yes", "on"}:
                    after_scene_graph_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
                    print(f"[ReplanDebug] after_scene_graph state={after_scene_graph_state.tolist()}")
                if current_datapoint is not None:
                    print(f"[Step {step_idx}] TAL datapoint actions: {list(getattr(current_datapoint, 'actions', []))}")
                print(f"[Step {step_idx}] Calling TAL planner...")
                try:
                    tal_result = tal_planner.plan_first_action(
                        args.prompt,
                        current_scene_graph,
                        start_node=current_datapoint,
                    )
                    latest_raw_subtask = tal_result.first_action_text
                    latest_raw_parsed_subtask = parse_tal_subtask(
                        tal_result.predicted_actions[0] if tal_result.predicted_actions else latest_raw_subtask
                    )
                    latest_parsed_subtask = derive_executable_subtask(latest_raw_parsed_subtask)
                    if latest_parsed_subtask is not None and latest_parsed_subtask.is_navigation:
                        nav_key = (latest_parsed_subtask.name.lower(), tuple(latest_parsed_subtask.args))
                        if nav_key in completed_navigation_subtasks:
                            print(
                                f"[Step {step_idx}] Skipping already completed navigation subtask: "
                                f"{latest_parsed_subtask.text}"
                            )
                            latest_parsed_subtask = None
                            latest_subtask = latest_raw_subtask
                        else:
                            latest_subtask = latest_parsed_subtask.text
                    else:
                        latest_subtask = latest_parsed_subtask.text if latest_parsed_subtask is not None else latest_raw_subtask
                    latest_fused_prompt = build_fused_prompt(args.prompt, latest_subtask)
                except Exception as exc:  # noqa: BLE001
                    tal_result = TALPlanResult(
                        status="Error",
                        first_action_text=None,
                        predicted_actions=[],
                        current_scene_graph_json=current_scene_graph,
                        goal_scene_graph_json=None,
                        error=str(exc),
                    )
                    latest_subtask = None
                    latest_parsed_subtask = None
                    latest_fused_prompt = args.prompt
                    latest_raw_subtask = None
                    latest_raw_parsed_subtask = None
                if os.environ.get("TAL_ONLINE_DEBUG_REPLAN", "0").lower() in {"1", "true", "yes", "on"}:
                    after_planner_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
                    print(f"[ReplanDebug] after_planner state={after_planner_state.tolist()}")
                force_replan = False
                print("=" * 80)
                print(f"[Step {step_idx}] user prompt: {args.prompt}")
                print(f"[Step {step_idx}] current scene graph: {json.dumps(current_scene_graph, ensure_ascii=False)}")
                print(f"[Step {step_idx}] TAL status: {tal_result.status}")
                print(f"[Step {step_idx}] TAL predicted actions(raw): {tal_result.predicted_actions}")
                print(f"[Step {step_idx}] TAL first action(raw text): {latest_raw_subtask}")
                print(f"[Step {step_idx}] TAL parsed subtask(raw): {latest_raw_parsed_subtask}")
                print(f"[Step {step_idx}] TAL execution subtask: {latest_subtask}")
                print(f"[Step {step_idx}] TAL parsed subtask(exec): {latest_parsed_subtask}")
                print(f"[Step {step_idx}] fused prompt: {latest_fused_prompt}")
                if tal_result.error:
                    print(f"[Step {step_idx}] TAL error: {tal_result.error}")

            if latest_parsed_subtask is not None and latest_parsed_subtask.is_navigation:
                derived_from = latest_parsed_subtask.raw.get("derived_from") if isinstance(latest_parsed_subtask.raw, Mapping) else None
                source_action_name = derived_from.get("name") if isinstance(derived_from, Mapping) else None
                nav_goal = build_navigation_goal(
                    runtime_ctx,
                    latest_parsed_subtask.args[0],
                    source_action_name=source_action_name,
                )
                print(
                    f"[Step {step_idx}] Routing TAL subtask to Nav2: "
                    f"{latest_parsed_subtask.text} -> goal(x={nav_goal.x:.3f}, y={nav_goal.y:.3f}, yaw={nav_goal.yaw:.3f})"
                )
                nav_bridge.sync_from_current_root()
                nav_bridge.set_active_goal(nav_goal)
                pending_nav = nav_client.send_goal(
                    nav_goal,
                    result_timeout=args.nav_goal_timeout_sec,
                    server_timeout=args.nav_server_timeout_sec,
                )
                deadline = time.monotonic() + args.nav_goal_timeout_sec
                while not pending_nav.done_event.wait(timeout=0.0):
                    if time.monotonic() >= deadline:
                        nav_client.cancel(pending_nav)
                        raise TimeoutError(
                            f"Timed out waiting for Nav2 goal after {args.nav_goal_timeout_sec:.1f}s: {nav_goal}"
                        )
                    advance_simulation(world, nav_bridge, args.nav_control_dt, render=not args.headless)
                    time.sleep(min(max(args.nav_control_dt * 0.25, 0.005), 0.02))

                if not pending_nav.success:
                    nav_bridge.set_active_goal(None)
                    accept_failed_goal = os.environ.get("TAL_NAV_ACCEPT_FAILED_GOAL_IF_CLOSE", "1").lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    close_pos_tol = max(float(os.environ.get("TAL_NAV_CLOSE_POSITION_TOL_M", "0.18")), 1e-3)
                    close_yaw_tol = max(float(os.environ.get("TAL_NAV_CLOSE_YAW_TOL_RAD", "0.80")), 1e-3)
                    if accept_failed_goal and navigation_goal_is_close_enough(
                        robot_root_controller,
                        nav_goal,
                        position_tolerance_m=close_pos_tol,
                        yaw_tolerance_rad=close_yaw_tol,
                    ):
                        print(
                            f"[Step {step_idx}] Nav2 returned failure status={pending_nav.status}, "
                            f"but robot is already close to goal; accept navigation result.",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(pending_nav.error or f"Nav2 navigation failed with status={pending_nav.status}")

                print(f"[Step {step_idx}] Nav2 goal reached successfully.", flush=True)
                nav_bridge.settle_to_goal_pose(nav_goal)
                nav_bridge.set_active_goal(None)
                completed_navigation_subtasks.add((latest_parsed_subtask.name.lower(), tuple(latest_parsed_subtask.args)))
                if isinstance(derived_from, Mapping):
                    latest_subtask = None
                    latest_fused_prompt = args.prompt
                    skip_replan_once = True
                    force_replan = False
                else:
                    force_replan = True
                latest_parsed_subtask = None
                step_idx += 1
                continue

            if policy is None:
                print("Connecting to OpenPI Policy Server...")
                policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
                print("Connected!")

            print(f"[Step {step_idx}] Sending fused prompt to OpenPI...")
            target_action = infer_action(policy, images, current_state, latest_fused_prompt)
            print(f"[Step {step_idx}] OpenPI first action: {target_action}")
            print(f"[Step {step_idx}] Applying action to robot...")
            apply_robot_action(
                robot,
                world,
                target_action,
                target_indices,
                ArticulationAction,
                nav_bridge=nav_bridge,
                dt=args.nav_control_dt,
                render=not args.headless,
            )
            step_idx += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Control loop failed: {exc}")
        traceback.print_exc()
    finally:
        print("[Shutdown] Closing TAL runtime and SimulationApp...")
        try:
            nav_bridge.close()
        except Exception:
            pass
        runtime_ctx.close()


if __name__ == "__main__":
    main()
