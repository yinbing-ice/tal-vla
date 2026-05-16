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

    def _refresh_live_datapoint(self) -> Any:
        isaac_env = self._runtime.isaac_env
        isaac_env.update_metrics()
        isaac_env.resetDatapoint(self._runtime.sim_env_config)
        isaac_env.initRootNode()
        # 2026-05-13 修改：在线 TAL 当前场景图不再直接读取 Isaac 上帝视角 datapoint，
        # 而是先生成真值 datapoint，再统一转换成 YOLO 观测版 datapoint。
        # 这样运行时重规划与 exploration / AFE / APN 的训练数据口径保持一致。
        return isaac_env.getObservedDatapoint(self._runtime.sim_env_config)

    def get_current_scene_graph(
        self,
        *,
        state_name: str | None = None,
        manual_scene_graph: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Any | None]:
        if manual_scene_graph is not None:
            return manual_scene_graph, None

        if state_name is None:
            datapoint = self._refresh_live_datapoint()
        else:
            self._runtime.isaac_env.update_metrics()
            datapoint = self._runtime.isaac_env.getObservedDatapoint(self._runtime.sim_env_config)

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


def capture_rgb_images(cam_high: Any, cam_wrist: Any) -> dict[str, np.ndarray]:
    img_high_rgba = cam_high.get_rgba()[:, :, :3]
    img_wrist_rgba = cam_wrist.get_rgba()[:, :, :3]

    if img_high_rgba.dtype == np.float32:
        img_high_rgb = (img_high_rgba * 255).astype(np.uint8)
        img_wrist_rgb = (img_wrist_rgba * 255).astype(np.uint8)
    else:
        img_high_rgb = img_high_rgba
        img_wrist_rgb = img_wrist_rgba

    return {
        "cam_high": cv2.cvtColor(img_high_rgb, cv2.COLOR_RGB2BGR),
        "cam_wrist": cv2.cvtColor(img_wrist_rgb, cv2.COLOR_RGB2BGR),
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


def apply_robot_action(robot: Any, target_action: np.ndarray, target_indices: np.ndarray, ArticulationAction: Any) -> None:
    action_cmd = ArticulationAction(
        joint_positions=target_action.astype(np.float32),
        joint_indices=target_indices.astype(np.int32),
    )
    robot.apply_action(action_cmd)


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
    start_positions = robot.get_joint_positions()[target_indices]
    num_steps = 240
    for i in range(num_steps):
        alpha = (i + 1) / float(num_steps)
        interpolated_positions = start_positions + alpha * (TRAIN_INIT_STATE - start_positions)
        step_action = ArticulationAction(
            joint_positions=interpolated_positions,
            joint_indices=target_indices.astype(np.int32),
        )
        robot.apply_action(step_action)
        world.step()

    final_action = ArticulationAction(
        joint_positions=TRAIN_INIT_STATE,
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

    world.reset()

    sim_dof_names = robot.dof_names
    target_indices = []
    for name in JOINT_NAMES_IN_ORDER:
        if name in sim_dof_names:
            target_indices.append(sim_dof_names.index(name))
        else:
            print(f"Warning: joint {name} was not found in simulation.")
    target_indices = np.array(target_indices, dtype=np.int32)

    smoothly_move_robot_root(robot, world, ROBOT_START_WORLD_POSITION)
    warm_up_robot(robot, world, target_indices, ArticulationAction)
    print("Starting TAL(native scene graph) + OpenPI closed-loop inference...")

    latest_subtask = None
    latest_fused_prompt = args.prompt
    step_idx = 0

    try:
        while True:
            print(f"[Loop] entering step {step_idx}")
            world.step()
            if args.max_steps >= 0 and step_idx >= args.max_steps:
                print("Reached max steps, exiting.")
                break

            print(f"[Step {step_idx}] Capturing RGB images...")
            images = capture_rgb_images(cam_high, cam_wrist)
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
                print(f"[Step {step_idx}] Building current scene graph from TAL runtime...")
                current_scene_graph, current_datapoint = scene_graph_provider.get_current_scene_graph(
                    state_name=scene_graph_state_name,
                    manual_scene_graph=manual_scene_graph,
                )
                if current_datapoint is not None:
                    print(f"[Step {step_idx}] TAL datapoint actions: {list(getattr(current_datapoint, 'actions', []))}")
                print(f"[Step {step_idx}] Calling TAL planner...")
                tal_result = tal_planner.plan_first_action(
                    args.prompt,
                    current_scene_graph,
                    start_node=current_datapoint,
                )
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
            apply_robot_action(robot, target_action, target_indices, ArticulationAction)
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
