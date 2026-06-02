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


CAMERA_HIGH_PATH = "/World/high"
CAMERA_WRIST_PATH = "/World/Mobie_grasper2/firefighter/joint6/wrist"
# 2026-05-24 修改：在线闭环里把 OpenPI 低层控制相机和 TAL 高层规划相机解耦。
# 默认仍保持 TAL 使用 high；如果场景里新增了 /World/high2，可通过 TAL_ONLINE_TAL_CAMERA_PATH=/World/high2 单独给 TAL/YOLO 使用。
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
    # 2026-05-18 修改：在部分 Isaac 版本里，articulation 在 reset 或根节点位姿改动后，
    # 低层控制器句柄可能需要重新初始化/后处理一次，否则后续 apply_action 看起来像“没生效”。
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
    # 2026-05-18 修改：在线闭环里默认优先“保住当前姿态”，
    # 因此先做更长一点的非 reset 句柄等待，不再默认一上来就走 reset fallback。
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

    # 2026-05-18 修改：只有显式允许时才启用 reset 兜底，
    # 避免为了拿句柄把用户手动摆好的机械臂/夹爪姿态重置掉，表现成“突然僵直”。
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
    # 2026-06-02 修改：Isaac 的 world.step(render=True) 往往会在函数内部先完成物理解算并渲染，
    # 如果 root guard 在 step 返回后才修正底盘，VNC/相机看到的仍可能是被物理解算顶歪的那一帧。
    # 因此有 root guard 时改成：不渲染地推进物理 -> 修正 root -> 再单独渲染。
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
    # 2026-05-18 修改：当 articulation 句柄还没 ready 时，直接从 live stage 的 joint prim 读取当前关节值，
    # 用来保存用户在 GUI 里手动摆好的预抓取姿态。这样即使后面为了拿句柄走 reset fallback，也有机会把姿态恢复回来。
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
    # 2026-05-18 修改：句柄初始化/兜底 reset 之后，把用户在场景里已经摆好的机械臂/夹爪姿态恢复回去，
    # 避免脚本进入 TAL + OpenPI 主循环前就把预抓取姿态“拉直/打掉”。
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
    # 2026-06-02 修改：Isaac 的 XFormPrim.get/set_world_poses 使用 wxyz 四元数顺序。
    # 之前这里额外转成 xyzw，写回 root orientation 时会把 yaw/roll/pitch 搞乱，
    # 视频里小车车头大幅翘起、车尾离地很符合这个错误的表现。
    return rpy_to_quaternion(roll, pitch, yaw)


def quaternion_wxyz_to_rpy(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> tuple[float, float, float]:
    # 2026-06-02 修改：和上面的写入函数保持一致，root orientation 统一按 Isaac wxyz 读取。
    if quaternion is None:
        return 0.0, 0.0, 0.0
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return 0.0, 0.0, 0.0
    return quaternion_to_rpy(q)


def quaternion_wxyz_to_yaw(quaternion: np.ndarray | list[float] | tuple[float, ...] | None) -> float:
    # 2026-06-02 修改：Nav2 close 判断也必须用同一套 wxyz 约定，否则会误判 yaw 误差。
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
        # 2026-05-30 修改：在线重规划只读取当前世界状态，不再重建 root datapoint，
        # 避免高层 TAL/YOLO 刷新动作干扰底层 OpenPI 连续控制。
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
    # 2026-05-18 修改：在线控制阶段仍然只使用 get_rgba() 这一条取图路径，
    # 但把“首帧等待”和“空帧重试”从直接崩溃改成温和等待，避免 step 0 因相机尚未出图退出。
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
    # 2026-05-31 修改：当前 TAL runtime 里注入的 isaac_env 是模块，不是类实例。
    # 旧版 moveTo 逻辑里误按实例接口调用 get_metrics()，会直接报 AttributeError。
    # 这里统一兼容当前模块式 API：先刷新模块级 metrics，再从模块全局状态里读取。
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
        positions, orientations = self._prim.get_world_poses()
        if positions is None or len(positions) == 0:
            return None, None
        position = np.asarray(positions[0], dtype=np.float32)
        orientation = None
        if orientations is not None and len(orientations) > 0:
            orientation = np.asarray(orientations[0], dtype=np.float32)
        return position, orientation

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
        positions = target_position.reshape(1, 3)
        orientations = None if target_orientation is None else target_orientation.reshape(1, 4)
        self._prim.set_world_poses(positions=positions, orientations=orientations)

    def set_linear_velocity(self, velocity: np.ndarray) -> None:
        # 2026-06-02 修改：刚体平移 root pose 前后主动清零 XFormPrim 速度。
        # 单靠 Articulation 的速度清零不一定覆盖根 prim，容易让 PhysX 残余角速度继续把车身抬起来。
        if hasattr(self._prim, "set_linear_velocities"):
            try:
                self._prim.set_linear_velocities(np.asarray(velocity, dtype=np.float32).reshape(1, 3))
            except Exception:
                pass

    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        # 2026-06-02 修改：同上，特别清零角速度，避免导航过程中姿态越积越歪。
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
        # 2026-06-02 修改：覆盖 nav_bridge 创建前的准备阶段。
        # 初始化句柄、reset fallback、恢复机械臂姿态都会推进 world.step；如果不在这些 step 后固定 root，
        # 小车可能在 TAL 调用前就已经被机械臂驱动/接触解算带偏，后续导航只是在错误姿态上继续移动。
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
        # 2026-06-01 修改：当前工程此前把导航 bridge 简化得太厉害，
        # 每次都从 PhysX 当前 root pose 读回位置再继续写回，导致悬空/姿态异常被持续放大。
        # 这里恢复接近老版本的“自维护运动状态”思路：
        # 单独维护逻辑导航位姿 (_x/_y/_yaw) 和写回 root 的显示位姿 (_xform_x/_xform_y/_xform_yaw)，
        # 避免把物理抖动再次反馈进导航积分。
        self._root_yaw_offset = float(os.environ.get("TAL_ROBOT_ROOT_YAW_OFFSET", "0.0"))
        self._max_linear_accel = max(float(os.environ.get("TAL_NAV_MAX_LINEAR_ACCEL", "0.20")), 1e-4)
        self._max_angular_accel = max(float(os.environ.get("TAL_NAV_MAX_ANGULAR_ACCEL", "0.50")), 1e-4)
        # 2026-05-31 修改：当前 moveTo 仍采用“刚体平移”过渡方案，为了尽量减小
        # PhysX 对 root 瞬移的敏感性，这里对每步平移和每步转角做硬限幅。
        # 后续若切到真实轮式控制，可删除这层保守约束。
        self._max_translation_step_m = max(float(os.environ.get("TAL_NAV_MAX_TRANSLATION_STEP_M", "0.01")), 1e-4)
        self._max_yaw_step_rad = max(float(os.environ.get("TAL_NAV_MAX_YAW_STEP_RAD", "0.03")), 1e-4)
        # 2026-05-31 修改：进一步降低 root pose 的写入频率。
        # 现在不是每个仿真步都 set_world_pose，而是累计到固定时间片再写一次，
        # 尽量减小对 PhysX broadphase / articulation 的持续刺激。
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
        # 2026-05-31 修改：记录导航桥初始化时的底盘根节点高度。
        # 目前 moveTo 仍采用“刚体平移”的过渡实现，不走真实轮子驱动；
        # 若每帧直接把 Isaac 当前返回的 z 再写回去，容易把物理解算中的抖动不断积累，
        # 最终出现小车“飞起来”。这里固定使用启动时的根节点高度来平移底盘。
        initial_position, _ = self._robot_root_controller.get_world_pose()
        self._base_z = float(initial_position[2]) if initial_position is not None else 0.0
        position, orientation = self._robot_root_controller.get_world_pose()
        if position is None:
            raise RuntimeError("Failed to read initial robot world pose for navigation bridge.")
        initial_root_roll, initial_root_pitch, initial_root_yaw = quaternion_wxyz_to_rpy(orientation)
        # 2026-06-02 修改：当前 moveTo 目标是“连续刚体在地面上平移”，不是让轮子真实滚动。
        # 因此导航阶段默认压平 root 的 roll/pitch，只保留 yaw；如果后续确认 USD 模型本身
        # 需要非零初始姿态，可设置 TAL_NAV_FLATTEN_ROOT_ATTITUDE=0 回退为保留初始 roll/pitch。
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
                # 2026-06-02 修改：root yaw offset 写回也使用 Isaac wxyz 顺序，
                # 并遵守上面的平面化策略，避免一开始就把底盘写成翘头姿态。
                orientation=rpy_to_quaternion_wxyz(initial_root_roll, initial_root_pitch, corrected_root_yaw),
            )
            self._zero_root_velocity()
            position, orientation = self._robot_root_controller.get_world_pose()
        self._xform_x = float(position[0])
        self._xform_y = float(position[1])
        self._xform_z = float(position[2])
        # 2026-06-02 修改：XFormPrim orientation 统一按 wxyz 解读；导航刚体移动默认只改 yaw。
        # 这样每次 root pose 写回都是固定 z + 水平姿态，避免把物理抖动反馈进下一步。
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
        # 2026-06-02 修改：每次刚体搬运前后都清零 root 和 articulation 速度。
        # 这是为了让“连续刚体平移”更像一个确定性的运动学更新，而不是让 PhysX 继续积分旧的倾覆速度。
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
        # 2026-06-02 修改：真正开始 moveTo 前，从 Isaac 当前 root 重新同步导航内部状态。
        # bridge 创建得比较早，warmup/TAL/相机预热之后，内部缓存的 _xform_* 可能还是旧位置；
        # 若直接 set_active_goal，就会先把车“闪现”回旧缓存。这里把导航起点改成当前实际位置，
        # 并按 flatten 策略只保留 yaw，让刚体移动从水平地面姿态开始。
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
        # 2026-06-02 修改：把底盘当作运动学基座写回，而不是让 PhysX 自由积分。
        # 这一步不仅用于导航，也用于 warmup/OpenPI 控制阶段，避免机械臂驱动反作用力把整车掀起来。
        self._zero_root_velocity()
        self._robot_root_controller.set_world_pose(
            position=np.array([self._xform_x, self._xform_y, self._xform_z], dtype=np.float32),
            orientation=self._root_orientation(),
        )
        self._zero_root_velocity()

    def enforce_root_pose_after_step(self) -> None:
        # 2026-06-02 修改：world.step() 之后再钉一次 root pose。
        # 如果只在 step 前写位姿，PhysX 仍可能在这一帧里通过接触/关节力把底盘顶歪。
        # 2026-06-02 修改：只在 moveTo 导航阶段钉 root pose。非导航阶段如果也强制写回，
        # TAL/OpenPI 切换前就会把小车拉回 bridge 初始化时的旧位置/旧姿态，表现为“闪现回原位”。
        if self._enforce_root_pose and (self._active_goal is not None or self._idle_root_hold_enabled):
            self._apply_root_pose()

    def enforce_after_step(self) -> None:
        # 2026-06-02 修改：和 RobotRootPoseGuard 对齐 step 后修正接口。
        # _step_world_with_root_guard 会统一调用 enforce_after_step；导航阶段这里转到
        # IsaacNavBridge 自己维护的 root pose，避免 AttributeError 让仿真提前退出。
        self.enforce_root_pose_after_step()

    @staticmethod
    def _step_towards(current: float, target: float, max_delta: float) -> float:
        if target > current:
            return min(current + max_delta, target)
        return max(current - max_delta, target)

    def _start_bridge_process(self) -> None:
        # 2026-05-30 修改：不要在 Isaac Python 进程内直接 import rclpy。
        # 当前机器的 ROS2 是独立的 Jazzy 环境，若把 rclpy 拉进 Isaac 解释器，
        # 很容易和 Kit/OmniGraph/Replicator 的 Python 运行时冲突并触发崩溃。
        # 因此桥接器改成真正的外部 ROS 子进程。
        # 2026-05-31 修改：桥接子进程若启动失败，直接把 stdout/stderr 带回主进程，
        # 避免只看到“exited immediately”而定位不到真实根因。
        self._bridge_stdout_path = Path(f"/tmp/isaac_nav_bridge_{self._state_port}.out.log")
        self._bridge_stderr_path = Path(f"/tmp/isaac_nav_bridge_{self._state_port}.err.log")
        cmd = [
            "bash",
            "-lc",
            (
                # 2026-05-31 修改：桥接 ROS 子进程必须主动清理 Isaac/Kit 注入的 Python 环境变量。
                # 否则 source Jazzy 后仍会错误地混用 Isaac 自带标准库，触发 SRE module mismatch，
                # 并进一步导致 rclpy 无法按 ROS2 的 Python 环境正常导入。
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
        # 2026-05-31 修改：只有在真正执行导航子任务时，才允许导航桥接管底盘平移。
        # 非导航阶段保持完全静默，避免 warmup / 相机预热 / OpenPI 控制阶段也误改 robot root。
        if self._active_goal is None:
            self._cmd_vx = 0.0
            self._cmd_vw = 0.0
            self._applied_vx = 0.0
            self._applied_vw = 0.0
            self._sim_time_s += dt
            self._root_update_accum_s = 0.0
            # 2026-06-02 修改：非导航阶段只发布 odom/clock 状态，不再持续写回 root pose。
            # 之前 bridge 一创建就把当时读到的 root pose 固化；TAL 规划、相机预热、OpenPI 连接前
            # 每步都 _apply_root_pose()，会表现成“小车闪现回原来的位置/姿态”。真正移动底盘只应发生在
            # moveTo active goal 期间，OpenPI pick/place 阶段也不应该被导航桥抢 root 控制权。
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
        # 2026-06-01 修改：root pose 写回时只使用 bridge 自维护的连续状态，不再读取
        # PhysX 当前 root pose 作为下一步输入，避免悬空/姿态异常被积分放大。
        # 2026-06-02 修改：写 root pose 前后清零速度，并用 wxyz 四元数 + 水平 roll/pitch，
        # 让小车按运动学刚体贴地移动，不再把当前倾斜姿态带进下一帧。
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
        # 2026-06-02 修改：导航结束强制落到同一套平面 root pose，避免 settle 时再次写错姿态。
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
                # 2026-05-31 修改：导航 goal 子进程和桥接子进程一样，需要和 Isaac Python 环境彻底隔离。
                # 这里显式清理 PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH，并强制使用系统 Python，
                # 避免 ROS2 Jazzy 的 rclpy 再次被 Isaac 解释器环境污染。
                # 2026-05-31 修改：当前 robot_ws 的 local_setup.bash 不能替代 ROS2 Jazzy 主环境，
                # 若只 source 工作区而不先 source /opt/ros/jazzy/setup.bash，rclpy 仍然找不到。
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
        # 2026-06-02 修改：相机预热属于 TAL/OpenPI 前的准备阶段，不应让非 active 的导航桥
        # 接管 root；若传入 early guard，则继续把底盘钉回启动时的地面姿态，避免预热 world.step 把车撬歪。
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
    # 2026-05-18 修改：进入主循环前做双目预热，不再只 sleep 干等。
    # 对 Isaac 相机来说，真正让 high / wrist 两路都产出首帧，通常需要推进几次渲染/仿真步。
    # 2026-05-24 修改：如果 TAL 规划单独使用 high2，也一并在这里预热首帧，
    # 这样 OpenPI 仍然用原来的 high+wrist，而 TAL/YOLO 可以稳定读到自己的高位相机。
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
    # 2026-05-24 修改：TAL 高层规划图像单独读取；如果未配置独立 high2，就回退复用 cam_high。
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

    # 2026-05-18 修改：这里优先使用 articulation 的 joint position target 接口，
    # 因为它比单次 apply_action 更像“持续保持目标”，在当前 Mobie_grasper2 上更稳定。
    # 如果当前 Isaac 版本没有这个接口，再回退到 apply_action。
    use_position_targets = hasattr(robot, "set_joint_position_targets")

    # 2026-05-18 修改：为了排查当前机器人 articulation 偶发“不响应目标”的问题，
    # 提供一个直接写入关节位置的调试兜底开关。
    # 这个模式不是物理驱动，而是直接把关节 teleport 到目标值，
    # 更适合先验证 TAL + OpenPI 闭环是否整体连通。
    use_direct_set = os.environ.get("TAL_ONLINE_DIRECT_SET_JOINTS", "0").lower() in {"1", "true", "yes", "on"}
    can_direct_set = hasattr(robot, "set_joint_positions")

    # 2026-05-18 修改：在线 TAL + OpenPI 闭环里，只在循环末尾瞬时下发一次目标时，
    # 会偶发出现“OpenPI 输出在变、关节状态几乎不动”的现象。
    # 这里把同一个关节目标在几个连续子步中重复下发，并同步推进 world.step，
    # 让控制器/drive 有足够时间真正吃进目标值。
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

    # =================================================================
    # 🌟 核心修复 1：防止遁地（穿模）
    # 强制将目标位置的 Z 轴高度，设为机器人当前安全的物理高度
    # =================================================================
    print(f"[InitMove] Original Target Z: {target_position[2]:.4f}, Overriding to Safe Z: {start_position[2]:.4f}")
    target_position[2] = start_position[2]

    print(f"[InitMove] Start world position: {start_position.tolist()}")
    print(f"[InitMove] Target world position: {target_position.tolist()}")

    for step in range(num_steps):
        alpha = float(step + 1) / float(num_steps)
        smooth_alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
        
        interpolated_position = start_position + smooth_alpha * (target_position - start_position)
        
        # =================================================================
        # 🌟 核心修复 2：防摩擦微悬浮
        # 在移动过程中 (类似于一个抛物线)，将机器人最高抬升 3 厘米 (0.03米)
        # 避免脚底板/轮子和地面产生物理碰撞和摩擦，导致机器人翻转
        # =================================================================
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

    # 稳定阶段：给物理引擎一点时间让机器人稳稳落回地面
    for _ in range(60):
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        advance_simulation(world, nav_bridge, dt, render=False)

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
    # 2026-06-02 修改：准备阶段默认直接写关节位置，而不是用 apply_action 通过物理驱动慢慢推。
    # 当前任务希望底盘按连续刚体运动保持在地面上，warmup 的机械臂驱动力会给底盘反作用力，
    # 视频里“一开始车尾翘起来”很像这个阶段把车体撬动；直接插值关节可避免准备阶段先把 root 带偏。
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
    # 2026-05-24 修改：OpenPI 继续使用 high+wrist；如果单独配置了 TAL 相机，则额外初始化 high2 给高层规划用。
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
    # 2026-06-02 修改：准备阶段也会推进仿真步，必须在初始化/恢复关节前就准备 root guard。
    # 否则小车可能在 TAL 规划前已经因为机械臂驱动反作用力漂移或翘头。
    robot_root_controller = RobotRootPoseController(XFormPrim(robot_prim_path))
    early_root_pose_guard = RobotRootPoseGuard(robot, robot_root_controller)

    # 2026-05-18 修改：在线闭环按“两阶段”处理启动逻辑：
    # 先尽量保存场景里当前已经摆好的机械臂/夹爪姿态，再做句柄初始化，最后把姿态恢复回去。
    # 这样既能拿到可用 articulation handle，又尽量不破坏预抓取位姿。
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

    # 2026-06-01 修改：导航阶段不要直接通过 Articulation 对象 set_world_pose 搬 robot root。
    # 老版本这里使用的是根 prim 的 XFormPrim 控制器，这样比直接改 articulation 更稳，
    # 能减少底盘悬空、关节状态变 nan、以及后续 PhysX broadphase 报错。

    sim_dof_names = robot.dof_names
    target_indices = []
    for name in JOINT_NAMES_IN_ORDER:
        if name in sim_dof_names:
            target_indices.append(sim_dof_names.index(name))
        else:
            print(f"Warning: joint {name} was not found in simulation.")
    target_indices = np.array(target_indices, dtype=np.int32)

    nav_bridge = IsaacNavBridge(robot, robot_root_controller)

    # 2026-05-30 修改：在线 moveTo 交给导航子模块执行，默认仍禁用 root move，
    # 避免直接搬 articulation 根节点时触发 PhysX 不稳定。
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
        # 2026-06-02 修改：Nav2/相机预热阶段还没有 active moveTo，继续使用 early guard。
        # 这样准备阶段不会因为 bridge 非 active 而失去 root 固定，也不会把旧 bridge 状态写回。
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
                # 2026-05-18 修改：主循环中若相机偶发空帧，不直接终止整条控制链，
                # 而是复用上一帧图像继续运行，优先保证 TAL + OpenPI + 物理控制主流程稳定。
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
                    # 2026-05-31 修改：当前 moveTo 仍是“Nav2 高层 + Isaac 刚体平移底盘”的过渡方案，
                    # Nav2 末段偶尔会返回 status=6（中止/失败），但底盘实际上已经被带到目标附近。
                    # 若此时直接抛异常，会把整条 move->pick->place 链打断；因此先检查是否已经足够接近目标，
                    # 接近则接受这次导航并继续后续抓取/放置。
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
