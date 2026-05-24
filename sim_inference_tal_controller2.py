from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import json
import os
from pathlib import Path
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
parser.add_argument("--headless", action="store_true", default=False, help="Run Isaac Sim in headless mode")
args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args


CAMERA_HIGH_PATH = "/World/high"
CAMERA_WRIST_PATH = "/World/Mobie_grasper2/firefighter/joint6/wrist"
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


def _initialize_robot_handles_if_needed(robot: Any, world: Any, *, headless: bool) -> None:
    # 2026-05-18 修改：在线闭环里默认优先“保住当前姿态”，
    # 因此先做更长一点的非 reset 句柄等待，不再默认一上来就走 reset fallback。
    init_steps = max(int(os.environ.get("TAL_ONLINE_ARTICULATION_INIT_STEPS", "30")), 1)
    for _ in range(init_steps):
        world.step(render=not headless)
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
        world.step(render=not headless)
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


def _restore_robot_pose_if_available(robot: Any, world: Any, pose_state: np.ndarray | None, pose_indices: np.ndarray | None, ArticulationAction: Any, *, headless: bool) -> None:
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
        world.step(render=not headless)


@dataclasses.dataclass
class TALPlanResult:
    status: str
    first_action_text: str | None
    predicted_actions: list[Any]
    current_scene_graph_json: dict[str, Any] | None = None
    goal_scene_graph_json: dict[str, Any] | None = None
    error: str | None = None


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

    def _refresh_live_datapoint(self, image_rgb: np.ndarray | None = None) -> Any:
        isaac_env = self._runtime.isaac_env
        isaac_env.update_metrics()
        # 2026-05-18 修改：在线控制阶段这里只读取当前仿真状态，不再像 exploration 那样
        # 反复 resetDatapoint/initRootNode 重建新的轨迹起点。
        # 同时优先复用主控制循环已经采集到的 cam_high 图像做 YOLO，避免重规划再触发一次相机读取。
        return isaac_env.getObservedDatapoint(self._runtime.sim_env_config, image_rgb=image_rgb)

    def get_current_scene_graph(
        self,
        *,
        state_name: str | None = None,
        manual_scene_graph: dict[str, Any] | None = None,
        image_rgb: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], Any | None]:
        if manual_scene_graph is not None:
            return manual_scene_graph, None

        if state_name is None:
            datapoint = self._refresh_live_datapoint(image_rgb=image_rgb)
        else:
            self._runtime.isaac_env.update_metrics()
            datapoint = self._runtime.isaac_env.getObservedDatapoint(
                self._runtime.sim_env_config,
                image_rgb=image_rgb,
            )

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


def wait_for_camera_frames(cam_high: Any, cam_wrist: Any, world: Any, *, headless: bool) -> dict[str, np.ndarray]:
    # 2026-05-18 修改：进入主循环前做双目预热，不再只 sleep 干等。
    # 对 Isaac 相机来说，真正让 high / wrist 两路都产出首帧，通常需要推进几次渲染/仿真步。
    preflight_retries = max(int(os.environ.get("TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES", "120")), 1)
    for _ in range(preflight_retries):
        world.step(render=not headless)
        high = _try_read_camera_rgb(cam_high)
        wrist = _try_read_camera_rgb(cam_wrist)
        if high is not None and wrist is not None:
            return {"cam_high": high, "cam_wrist": wrist}
        time.sleep(0.02)
    raise RuntimeError(
        f"Camera preflight failed after {preflight_retries} retries; cam_high/cam_wrist did not both produce valid frames"
    )


def capture_rgb_images(cam_high: Any, cam_wrist: Any) -> dict[str, np.ndarray]:
    return {
        "cam_high": _read_camera_rgb(cam_high, "cam_high"),
        "cam_wrist": _read_camera_rgb(cam_wrist, "cam_wrist"),
    }


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
        world.step(render=not args.headless)

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
        world.step(render=False)

    final_position, _ = robot.get_world_pose()
    if final_position is None:
        raise RuntimeError("Failed to verify final world pose for robot")
    print(f"[InitMove] Final world position: {np.asarray(final_position, dtype=np.float32).tolist()}")


def warm_up_robot(robot: Any, world: Any, target_indices: np.ndarray, ArticulationAction: Any) -> None:
    target_state = _get_initial_robot_state()
    start_positions = robot.get_joint_positions()[target_indices]
    num_steps = 240
    for i in range(num_steps):
        alpha = (i + 1) / float(num_steps)
        interpolated_positions = start_positions + alpha * (target_state - start_positions)
        step_action = ArticulationAction(
            joint_positions=interpolated_positions,
            joint_indices=target_indices.astype(np.int32),
        )
        robot.apply_action(step_action)
        world.step()

    final_action = ArticulationAction(
        joint_positions=target_state,
        joint_indices=target_indices.astype(np.int32),
    )
    for _ in range(60):
        robot.apply_action(final_action)
        world.step(render=False)


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
    cam_high.initialize()
    cam_wrist.initialize()

    print("Connecting to OpenPI Policy Server...")
    policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    print("Connected!")

    tal_planner = LazyTALPlanner(runtime_ctx)
    scene_graph_provider = TALSceneGraphProvider(runtime_ctx)
    manual_scene_graph = load_manual_scene_graph(args.manual_scene_graph_json)

    saved_pose_state, saved_pose_indices = _capture_current_robot_pose(robot, robot_prim_path)

    # 2026-05-18 修改：在线闭环按“两阶段”处理启动逻辑：
    # 先尽量保存场景里当前已经摆好的机械臂/夹爪姿态，再做句柄初始化，最后把姿态恢复回去。
    # 这样既能拿到可用 articulation handle，又尽量不破坏预抓取位姿。
    if _should_reset_world():
        world.reset()
    _initialize_robot_handles_if_needed(robot, world, headless=args.headless)
    if _should_reinitialize_robot():
        _reinitialize_robot_if_possible(robot)
    _restore_robot_pose_if_available(
        robot,
        world,
        saved_pose_state,
        saved_pose_indices,
        ArticulationAction,
        headless=args.headless,
    )

    sim_dof_names = robot.dof_names
    target_indices = []
    for name in JOINT_NAMES_IN_ORDER:
        if name in sim_dof_names:
            target_indices.append(sim_dof_names.index(name))
        else:
            print(f"Warning: joint {name} was not found in simulation.")
    target_indices = np.array(target_indices, dtype=np.int32)

    # 2026-05-18 修改：默认不再主动搬动机器人根节点，先尽量保持和 sim_inference3.py 的稳定控制路径一致。
    # 之前这里的 root move 更像是当前连续物理控制失效的高风险点；需要时可用环境变量显式打开。
    if _should_move_robot_root():
        smoothly_move_robot_root(robot, world, ROBOT_START_WORLD_POSITION)
        if _should_reinitialize_robot():
            _reinitialize_robot_if_possible(robot)
    # 2026-05-18 修改：默认不再强制把机械臂拉回固定 TRAIN_INIT_STATE，
    # 以免覆盖用户在启动前手动调好的初始姿态；需要时可显式设置 TAL_ONLINE_WARMUP_ROBOT=1 打开。
    if _should_warmup_robot():
        warm_up_robot(robot, world, target_indices, ArticulationAction)
        if _should_reinitialize_robot():
            _reinitialize_robot_if_possible(robot)
    print("Starting TAL(native scene graph) + OpenPI closed-loop inference...")

    latest_subtask = None
    latest_fused_prompt = args.prompt
    latest_images = wait_for_camera_frames(cam_high, cam_wrist, world, headless=args.headless)
    step_idx = 0

    try:
        while True:
            print(f"[Loop] entering step {step_idx}")
            # 2026-05-18 修改：循环开头保留一次 world.step，用于刷新相机/仿真缓存；
            # 真正执行关节控制的主步进已下沉到 apply_robot_action 里做连续子步。
            world.step(render=not args.headless)
            if args.max_steps >= 0 and step_idx >= args.max_steps:
                print("Reached max steps, exiting.")
                break

            print(f"[Step {step_idx}] Capturing RGB images...")
            try:
                images = capture_rgb_images(cam_high, cam_wrist)
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

            if should_replan(step_idx, args.replan_every_n_steps):
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
                    image_rgb=cv2.cvtColor(images["cam_high"], cv2.COLOR_BGR2RGB),
                )
                if os.environ.get("TAL_ONLINE_DEBUG_REPLAN", "0").lower() in {"1", "true", "yes", "on"}:
                    after_scene_graph_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
                    print(f"[ReplanDebug] after_scene_graph state={after_scene_graph_state.tolist()}")
                if current_datapoint is not None:
                    print(f"[Step {step_idx}] TAL datapoint actions: {list(getattr(current_datapoint, 'actions', []))}")
                print(f"[Step {step_idx}] Calling TAL planner...")
                tal_result = tal_planner.plan_first_action(
                    args.prompt,
                    current_scene_graph,
                    start_node=current_datapoint,
                )
                if os.environ.get("TAL_ONLINE_DEBUG_REPLAN", "0").lower() in {"1", "true", "yes", "on"}:
                    after_planner_state, _, _ = read_robot_state(robot, JOINT_NAMES_IN_ORDER)
                    print(f"[ReplanDebug] after_planner state={after_planner_state.tolist()}")
                latest_subtask = tal_result.first_action_text
                latest_fused_prompt = build_fused_prompt(args.prompt, latest_subtask)
                print("=" * 80)
                print(f"[Step {step_idx}] user prompt: {args.prompt}")
                print(f"[Step {step_idx}] current scene graph: {json.dumps(current_scene_graph, ensure_ascii=False)}")
                print(f"[Step {step_idx}] TAL status: {tal_result.status}")
                print(f"[Step {step_idx}] TAL predicted actions(raw): {tal_result.predicted_actions}")
                print(f"[Step {step_idx}] TAL first action(text): {latest_subtask}")
                print(f"[Step {step_idx}] fused prompt: {latest_fused_prompt}")
                if tal_result.error:
                    print(f"[Step {step_idx}] TAL error: {tal_result.error}")

            print(f"[Step {step_idx}] Sending fused prompt to OpenPI...")
            target_action = infer_action(policy, images, current_state, latest_fused_prompt)
            print(f"[Step {step_idx}] OpenPI first action: {target_action}")
            print(f"[Step {step_idx}] Applying action to robot...")
            apply_robot_action(robot, world, target_action, target_indices, ArticulationAction)
            step_idx += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Control loop failed: {exc}")
        traceback.print_exc()
    finally:
        print("[Shutdown] Closing TAL runtime and SimulationApp...")
        runtime_ctx.close()


if __name__ == "__main__":
    main()
