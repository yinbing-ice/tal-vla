import os
from copy import deepcopy
import numpy as np

# ==============================================================================
# 第一步：必须最先导入并实例化 SimulationApp
# ==============================================================================
try:
    from isaacsim import SimulationApp
except ModuleNotFoundError:
    from omni.isaac.kit import SimulationApp

_headless_env = os.environ.get("TAL_ISAAC_HEADLESS", "1").lower()
_headless = _headless_env not in {"0", "false", "no"}

# !!! 核心修改：在导入任何其他 Isaac Sim 模块前，必须先启动 SimulationApp !!!
simulation_app = SimulationApp({"headless": _headless})


# ==============================================================================
# 第二步：SimulationApp 启动后，才可以安全导入其它的 omni, isaacsim, pxr 模块
# ==============================================================================
try:
    import isaaclab.sim as sim_utils
    _HAS_ISAACLAB = True
except ModuleNotFoundError:
    sim_utils = None
    _HAS_ISAACLAB = False

try:
    from isaacsim.core.prims import XFormPrim
except ModuleNotFoundError:
    try:
        from omni.isaac.core.prims import XFormPrim
    except ModuleNotFoundError:
        from isaacsim.core.experimental.prims import XformPrim as XFormPrim

try:
    from isaacsim.core.api.simulation_context import SimulationContext
except ModuleNotFoundError:
    from omni.isaac.core import SimulationContext

try:
    from isaacsim.core.utils.stage import get_current_stage, open_stage, update_stage
except ModuleNotFoundError:
    from omni.isaac.core.utils.stage import get_current_stage, open_stage, update_stage

from pxr import Usd, UsdGeom

# ==============================================================================
# 第三步：导入本地自定义模块
# ==============================================================================
from src.envs.datapoint import Datapoint
from src.envs.perception_state import apply_yolo_observation_to_datapoint


sim = None
stage = None
object_prims = {}
datapoint = None
metrics = {}
constraints = {"husky": []}
config_cache = None
_logical_robot_metric = None
_logical_robot_marker_prim = None
_scene_summary_printed = False
_loaded_scene_usd_path = None
_scene_baseline_metrics = None
_scene_baseline_constraints = {"husky": []}

sticky, fixed, on, fueled, cut = [], [], [], [], []
cleaner, stick, clean, drilled, welded, painted = False, False, [], [], [], []


def _ensure_sim():
    global sim
    if sim is None:
        if _HAS_ISAACLAB:
            sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu")
            sim = sim_utils.SimulationContext(sim_cfg)
        else:
            sim = SimulationContext(stage_units_in_meters=1.0)
            sim.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    return sim


def _step(render=None, steps=1):
    skip_step_env = os.environ.get("TAL_ISAAC_SKIP_SIM_STEP", "0").lower()
    if skip_step_env in {"1", "true", "yes", "on"}:
        # 2026-05-11 修改：YOLO/Replicator 抓图后 Isaac 的 sim.step 偶发卡住。
        # 当前 TAL 符号后端主要通过 set_world_pose 直接更新物体位姿，探索阶段可只刷新 USD stage，
        # 避免 moveTo/pick 等高层动作卡在物理步进里。
        try:
            update_stage()
        except Exception:
            pass
        return

    sim_ctx = _ensure_sim()
    should_render = (not _headless) if render is None else render
    for _ in range(steps):
        sim_ctx.step(render=should_render)


def _open_stage(usd_path):
    global stage
    _ensure_sim()
    if not open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    update_stage()
    stage = get_current_stage()
    _ensure_sim().reset()
    _step(steps=2)
    return stage


def _scene_ready_for_reuse(config):
    return (
        _loaded_scene_usd_path == config.scene_usd_path
        and stage is not None
        and len(object_prims) == len(config.all_objects)
        and all(prim.is_valid() for prim in object_prims.values())
        and _scene_baseline_metrics is not None
    )


def _restore_metrics_snapshot(saved_metrics, saved_constraints=None):
    global constraints
    if saved_constraints is None:
        saved_constraints = {"husky": []}
    constraints = deepcopy(saved_constraints)

    for tal_name, prim in object_prims.items():
        if prim.is_valid() and tal_name in saved_metrics:
            pos, rot = saved_metrics[tal_name]
            if tal_name == "husky" and _use_logical_robot_motion():
                # 2026-05-17 修改：restore/回滚时也只恢复小车的 TAL 逻辑坐标，
                # 避免随机扩展 graph 节点时再次搬动 Mobie_grasper2 根节点导致 GUI 视觉体消失。
                _set_robot_metric(pos, rot)
                continue
            _set_world_pose(prim, np.array(pos), np.array(rot))

    _step(steps=2)
    update_metrics()


def _reset_runtime_state(config):
    global datapoint, metrics, constraints
    global sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted

    metrics = {}
    globals()["_logical_robot_metric"] = None
    globals()["_logical_robot_marker_prim"] = None
    constraints = {"husky": []}
    sticky, fixed, on, fueled, cut = [], [], [], [], []
    cleaner, stick, clean, drilled, welded, painted = False, False, [], [], [], []

    datapoint = Datapoint(config)
    datapoint.world = config.graph_world_name
    datapoint.goal = config.exploration_goal_name


def _initialize_scene_if_needed(config):
    global _loaded_scene_usd_path, _scene_baseline_metrics, _scene_baseline_constraints

    if _scene_ready_for_reuse(config):
        _restore_metrics_snapshot(_scene_baseline_metrics, _scene_baseline_constraints)
        return

    _open_stage(config.scene_usd_path)
    _refresh_scene_cache(config)
    update_metrics()

    _loaded_scene_usd_path = config.scene_usd_path
    _scene_baseline_metrics = deepcopy(metrics)
    _scene_baseline_constraints = {"husky": []}


def _compute_size(usd_prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    bbox = bbox_cache.ComputeWorldBound(usd_prim)
    bbox_range = bbox.GetRange()
    if bbox_range.IsEmpty():
        return [0.1, 0.1, 0.1]
    b_min = bbox_range.GetMin()
    b_max = bbox_range.GetMax()
    return [
        max(abs(b_max[0] - b_min[0]), 0.01),
        max(abs(b_max[1] - b_min[1]), 0.01),
        max(abs(b_max[2] - b_min[2]), 0.01),
    ]


def _read_prim_metadata(config, tal_name, usd_name):
    usd_prim = stage.GetPrimAtPath(f"/World/{usd_name}")
    if not usd_prim.IsValid():
        raise RuntimeError(f"Prim not found in expff.usd: /World/{usd_name}")
    metadata = {
        "tal_name": tal_name,
        "usd_name": usd_name,
        "prim_path": str(usd_prim.GetPath()),
        "type_name": usd_prim.GetTypeName(),
        "applied_schemas": list(usd_prim.GetAppliedSchemas()),
        "authored_attributes": sorted(attr.GetName() for attr in usd_prim.GetAuthoredAttributes()),
        "semantic_properties": list(config.object_property_map.get(tal_name, [])),
        "size": _compute_size(usd_prim),
    }
    return metadata


def _refresh_scene_cache(config):
    global object_prims
    object_prims = {}
    for tal_name, usd_name in config.tal_to_usd.items():
        prim_path = f"/World/{usd_name}"
        prim = XFormPrim(prim_path)
        if not prim.is_valid():
            raise RuntimeError(f"Invalid prim wrapper for: {prim_path}")
        object_prims[tal_name] = prim

        metadata = _read_prim_metadata(config, tal_name, usd_name)
        config.usd_metadata[tal_name] = metadata

        obj_entry = config.get_object_entry(tal_name)
        if obj_entry is not None:
            obj_entry["properties"] = list(metadata["semantic_properties"])
            obj_entry["size"] = list(metadata["size"])


def _print_scene_summary(config):
    global _scene_summary_printed
    summary_mode = os.environ.get("TAL_ISAAC_SCENE_SUMMARY", "once").strip().lower()
    if summary_mode in {"0", "false", "no", "off", "never"}:
        return
    if summary_mode not in {"always", "once"}:
        summary_mode = "once"
    if summary_mode == "once" and _scene_summary_printed:
        return

    print("--" * 20)
    print("Loaded expff.usd scene summary")
    for tal_name in config.all_objects:
        meta = config.usd_metadata.get(tal_name, {})
        obj_pos = metrics.get(tal_name, [[None, None, None], None])[0]
        print(
            f"{tal_name:>11} | path={meta.get('prim_path', 'N/A')} "
            f"| pos={obj_pos} | size={meta.get('size', [])} "
            f"| properties={meta.get('semantic_properties', [])}"
        )
    print("--" * 20)
    _scene_summary_printed = True


def _held_object(curr_constraints):
    held = curr_constraints.get("husky", [])
    if isinstance(held, (list, tuple)) and len(held) > 0:
        return held[0]
    return None


def get_held_object(curr_constraints=None):
    return _held_object(constraints if curr_constraints is None else curr_constraints)


def update_metrics():
    global metrics
    for tal_name, prim in object_prims.items():
        if prim.is_valid():
            pos, rot = prim.get_world_poses()
            metrics[tal_name] = [pos[0].tolist(), rot[0].tolist()]
    if _logical_robot_metric is not None:
        # 2026-05-17 修改：GUI/skip-step 探索时不直接搬动 Mobie_grasper2 的 USD 根节点，
        # 避免复杂 articulation 在 Isaac 视窗里出现视觉体消失；TAL graph 仍使用逻辑小车坐标。
        metrics["husky"] = deepcopy(_logical_robot_metric)
    return metrics


def _set_world_pose(prim, position, orientation=None):
    positions = np.asarray(position, dtype=np.float32).reshape(1, 3)
    orientations = None
    if orientation is not None:
        orientations = np.asarray(orientation, dtype=np.float32).reshape(1, 4)
    prim.set_world_poses(positions=positions, orientations=orientations)


def _show_logical_robot_marker():
    raw = os.environ.get("TAL_ISAAC_SHOW_LOGICAL_ROBOT_MARKER")
    if raw is None:
        return _use_logical_robot_motion() and not _headless
    return raw.lower() in {"1", "true", "yes", "on"}


def _ensure_logical_robot_marker():
    global _logical_robot_marker_prim
    if not _show_logical_robot_marker() or stage is None:
        return None
    marker_path = "/World/TAL_LogicalHuskyMarker"
    marker_usd_prim = stage.GetPrimAtPath(marker_path)
    if not marker_usd_prim.IsValid():
        # 2026-05-17 修改：真实 Mobie_grasper2 在 skip-step/GUI 下保持原地，
        # 用这个红色小球显示 TAL graph 中的逻辑小车位置，便于观察 exploration 是否在移动。
        sphere = UsdGeom.Sphere.Define(stage, marker_path)
        sphere.GetRadiusAttr().Set(0.16)
        sphere.GetDisplayColorAttr().Set([(1.0, 0.05, 0.02)])
        marker_usd_prim = sphere.GetPrim()
    if _logical_robot_marker_prim is None or not _logical_robot_marker_prim.is_valid():
        _logical_robot_marker_prim = XFormPrim(marker_path)
    return _logical_robot_marker_prim


def _update_logical_robot_marker(position):
    marker = _ensure_logical_robot_marker()
    if marker is None:
        return
    marker_pos = np.asarray(position, dtype=np.float32).copy()
    marker_pos[2] += 0.45
    _set_world_pose(marker, marker_pos)
    try:
        update_stage()
    except Exception:
        pass


def _use_logical_robot_motion():
    raw = os.environ.get("TAL_ISAAC_LOGICAL_ROBOT_MOTION")
    if raw is not None:
        return raw.lower() in {"1", "true", "yes", "on"}
    return os.environ.get("TAL_ISAAC_SKIP_SIM_STEP", "0").lower() in {"1", "true", "yes", "on"}


def _set_robot_metric(position, orientation=None):
    global _logical_robot_metric
    if orientation is None:
        orientation = metrics.get("husky", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])[1]
    _logical_robot_metric = [list(np.asarray(position, dtype=float)), list(orientation)]
    metrics["husky"] = deepcopy(_logical_robot_metric)
    _update_logical_robot_marker(position)


def _get_obj_size(config, obj_name):
    obj_entry = config.get_object_entry(obj_name)
    if obj_entry is None:
        return [0.1, 0.1, 0.1]
    return obj_entry["size"]


def _move_robot_near(target_name, approach_distance=None):
    robot_prim = object_prims["husky"]
    robot_pos = np.array(metrics["husky"][0], dtype=float)
    target_pos = np.array(metrics[target_name][0], dtype=float)
    delta = target_pos[:2] - robot_pos[:2]
    dist = np.linalg.norm(delta)
    if approach_distance is None:
        approach_distance = getattr(config_cache, "base_approach_distance", 0.50)
    approach_distance = max(float(approach_distance), 0.0)
    new_robot_pos = target_pos.copy()
    if dist < 1e-6:
        new_robot_pos[0] = target_pos[0] - approach_distance
    else:
        direction = delta / dist
        new_robot_pos[:2] = target_pos[:2] - direction * approach_distance
    new_robot_pos[2] = robot_pos[2]
    if _use_logical_robot_motion():
        # 2026-05-17 修改：探索/调试时小车只更新 TAL 逻辑坐标，不移动 Isaac 里的机器人模型。
        # 物体仍按动作结果移动，因此生成 graph 不受影响；GUI 中也能持续看到小车本体。
        _set_robot_metric(new_robot_pos)
    else:
        _set_world_pose(robot_prim, new_robot_pos)
    _step()
    update_metrics()


def _grounded_height(obj_name):
    obj_size = _get_obj_size(config_cache, obj_name)
    default_metric = [[0.0, 0.0, obj_size[2] / 2.0], [0.0, 0.0, 0.0, 1.0]]
    base_metric = config_cache.initial_object_metrics.get(obj_name, default_metric)
    base_z = base_metric[0][2]
    return max(base_z, obj_size[2] / 2.0 + 0.01)


def _place_on_target(obj_name, target_name):
    obj_size = _get_obj_size(config_cache, obj_name)
    target_size = _get_obj_size(config_cache, target_name)
    target_pos = np.array(metrics[target_name][0], dtype=float)

    place_pos = target_pos.copy()
    if target_name in config_cache.property2Objects["Container"]:
        place_pos[2] = target_pos[2] + max(target_size[2] * 0.15, obj_size[2] / 2.0 + 0.01)
    else:
        place_pos[2] = target_pos[2] + target_size[2] / 2.0 + obj_size[2] / 2.0 + 0.02

    _set_world_pose(object_prims[obj_name], place_pos)
    _step()
    update_metrics()


def start(config, input_args=None, INPUT_DATAPOINT=None):
    global stage, datapoint, metrics, constraints
    global sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted
    global config_cache

    config_cache = config
    _reset_runtime_state(config)

    if not os.path.exists(config.scene_usd_path):
        raise FileNotFoundError(f"USD scene does not exist: {config.scene_usd_path}")

    _initialize_scene_if_needed(config)
    config.initial_object_metrics = deepcopy(_scene_baseline_metrics)

    if INPUT_DATAPOINT is not None:
        input_metrics = None
        input_constraints = {"husky": []}
        for snapshot in getattr(INPUT_DATAPOINT, "metrics", []):
            if isinstance(snapshot, dict) and len(snapshot) != 0:
                input_metrics = deepcopy(snapshot)
                break
        for snapshot in getattr(INPUT_DATAPOINT, "constraints", []):
            if isinstance(snapshot, dict):
                input_constraints = deepcopy(snapshot)
                input_constraints.setdefault("husky", [])
                break
        if input_metrics is not None:
            _restore_metrics_snapshot(input_metrics, input_constraints)

    _print_scene_summary(config)

    r_pos = metrics["husky"][0]
    datapoint.addPoint(
        [r_pos[0], r_pos[1], r_pos[2], 0.0],
        sticky,
        fixed,
        cleaner,
        "Initialize",
        deepcopy(constraints),
        deepcopy(metrics),
        on,
        clean,
        stick,
        welded,
        drilled,
        painted,
        fueled,
        cut,
    )
    datapoint.addPoint(
        [r_pos[0], r_pos[1], r_pos[2], 0.0],
        sticky,
        fixed,
        cleaner,
        "Start",
        deepcopy(constraints),
        deepcopy(metrics),
        on,
        clean,
        stick,
        welded,
        drilled,
        painted,
        fueled,
        cut,
    )


def saveState():
    update_metrics()
    return deepcopy(metrics), deepcopy(constraints)


def restoreState(state_id, previous_constraints, previous_state_datapoint=None):
    global constraints
    global sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted

    saved_metrics = deepcopy(state_id)
    constraints = deepcopy(previous_constraints)

    for tal_name, prim in object_prims.items():
        if prim.is_valid() and tal_name in saved_metrics:
            pos, rot = saved_metrics[tal_name]
            if tal_name == "husky" and _use_logical_robot_motion():
                # 2026-05-17 修改：restore/回滚时也只恢复小车的 TAL 逻辑坐标，
                # 避免随机扩展 graph 节点时再次搬动 Mobie_grasper2 根节点导致 GUI 视觉体消失。
                _set_robot_metric(pos, rot)
                continue
            _set_world_pose(prim, np.array(pos), np.array(rot))

    _step(steps=2)
    update_metrics()

    if previous_state_datapoint is not None and "End" in previous_state_datapoint.actions:
        data_index = (
            len(previous_state_datapoint.actions)
            - 1
            - previous_state_datapoint.actions[::-1].index("End")
        )
        sticky = deepcopy(previous_state_datapoint.sticky[data_index])
        fixed = deepcopy(previous_state_datapoint.fixed[data_index])
        cleaner = deepcopy(previous_state_datapoint.cleaner[data_index])
        on = deepcopy(previous_state_datapoint.on[data_index])
        clean = deepcopy(previous_state_datapoint.clean[data_index])
        stick = deepcopy(previous_state_datapoint.stick[data_index])
        welded = deepcopy(previous_state_datapoint.welded[data_index])
        drilled = deepcopy(previous_state_datapoint.drilled[data_index])
        painted = deepcopy(previous_state_datapoint.painted[data_index])
        fueled = deepcopy(previous_state_datapoint.fueled[data_index])
        cut = deepcopy(previous_state_datapoint.cut[data_index])

    return saved_metrics


def execute_collect_data(config, symbolic_actions, goal_file=None, saveImg=False):
    global datapoint, constraints

    action = symbolic_actions["actions"][0]
    act_name = action["name"]
    args = list(action["args"])
    datapoint.addSymbolicAction(symbolic_actions["actions"])

    try:
        if act_name == "moveTo":
            _move_robot_near(args[0], config.base_approach_distance)

        elif act_name == "pick":
            target_obj = args[0]
            if target_obj not in config.property2Objects["Movable"]:
                raise Exception(f"Logical Rule: Cannot pick {target_obj}.")
            if _held_object(constraints) is not None:
                raise Exception("Logical Rule: Gripper is not free.")

            _move_robot_near(target_obj, config.pick_approach_distance)
            robot_pos = np.array(metrics["husky"][0], dtype=float)
            obj_size = _get_obj_size(config, target_obj)
            lift_pos = robot_pos.copy()
            lift_pos[2] = robot_pos[2] + max(obj_size[2], 0.2)
            _set_world_pose(object_prims[target_obj], lift_pos)
            constraints["husky"] = [target_obj]
            _step()
            update_metrics()

        elif act_name == "drop":
            requested_obj = args[0]
            held_obj = _held_object(constraints)
            if held_obj is None:
                raise Exception("Logical Rule: Gripper is free.")
            if held_obj != requested_obj:
                raise Exception(f"Logical Rule: Gripper holds {held_obj}, not {requested_obj}.")

            robot_pos = np.array(metrics["husky"][0], dtype=float)
            obj_size = _get_obj_size(config, held_obj)
            drop_pos = robot_pos.copy()
            drop_pos[2] = _grounded_height(held_obj)
            _set_world_pose(object_prims[held_obj], drop_pos)
            constraints["husky"] = []
            _step()
            update_metrics()

        elif act_name == "pushTo":
            obj_a, obj_b = args
            if obj_a not in config.property2Objects["Movable"]:
                raise Exception(f"Logical Rule: Cannot push {obj_a}.")
            if obj_b not in config.place_targets:
                raise Exception(f"Logical Rule: Unsupported push target {obj_b}.")

            _move_robot_near(obj_a, config.pick_approach_distance)
            _move_robot_near(obj_b, config.push_approach_distance)
            _place_on_target(obj_a, obj_b)

        elif act_name == "pickNplaceAonB":
            obj_a, obj_b = args
            if obj_a not in config.property2Objects["Movable"]:
                raise Exception(f"Logical Rule: Cannot pick and place {obj_a}.")
            if obj_b not in config.place_targets:
                raise Exception(f"Logical Rule: Cannot place on {obj_b}.")
            if obj_a == obj_b:
                raise Exception("Logical Rule: Cannot place object on itself.")

            _move_robot_near(obj_a, config.pick_approach_distance)
            constraints["husky"] = [obj_a]
            _move_robot_near(obj_b, config.place_approach_distance)
            _place_on_target(obj_a, obj_b)
            constraints["husky"] = []

        else:
            raise Exception(f"Error: Unsupported high-level action {act_name}.")

        update_metrics()
        robot_pos_final = metrics["husky"][0]
        datapoint.addPoint(
            [robot_pos_final[0], robot_pos_final[1], robot_pos_final[2], 0.0],
            sticky,
            fixed,
            cleaner,
            action,
            deepcopy(constraints),
            deepcopy(metrics),
            on,
            clean,
            stick,
            welded,
            drilled,
            painted,
            fueled,
            cut,
        )
        datapoint.addPoint(
            [robot_pos_final[0], robot_pos_final[1], robot_pos_final[2], 0.0],
            sticky,
            fixed,
            cleaner,
            "End",
            deepcopy(constraints),
            deepcopy(metrics),
            on,
            clean,
            stick,
            welded,
            drilled,
            painted,
            fueled,
            cut,
        )
        return True

    except Exception as e:
        datapoint.addSymbolicAction("Error = " + str(e))
        datapoint.addPoint(
            None,
            None,
            None,
            None,
            "Error = " + str(e),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        raise e


def initRootNode():
    update_metrics()
    r_pos = metrics["husky"][0]
    datapoint.addPoint(
        [r_pos[0], r_pos[1], r_pos[2], 0.0],
        sticky,
        fixed,
        cleaner,
        "End",
        deepcopy(constraints),
        deepcopy(metrics),
        on,
        clean,
        stick,
        welded,
        drilled,
        painted,
        fueled,
        cut,
    )


def getDatapoint(config, RESET_DATAPOINT=False):
    tmp_dp = datapoint.deepcopy()
    if RESET_DATAPOINT:
        resetDatapoint(config)
    return tmp_dp


def getObservedDatapoint(config, RESET_DATAPOINT=False, image_rgb=None):
    # 2026-05-13 修改：在线 TAL 重规划阶段需要和 YOLO 版训练图保持一致，
    # 因此这里提供统一接口，把 Isaac 真值 datapoint 转成“YOLO 观测 datapoint”后再返回。
    # 回滚和物理世界恢复仍然使用真值状态，不受这里影响。
    tmp_dp = datapoint.deepcopy()
    observed_dp = apply_yolo_observation_to_datapoint(
        config,
        tmp_dp,
        true_metrics=deepcopy(metrics),
        constraints=deepcopy(constraints),
        image_rgb=image_rgb,
    )
    if RESET_DATAPOINT:
        resetDatapoint(config)
    return observed_dp


def resetDatapoint(config, previous_state_datapoint=None):
    global datapoint
    datapoint = Datapoint(config)
    datapoint.world = config.graph_world_name
    datapoint.goal = config.exploration_goal_name
    if previous_state_datapoint is not None:
        datapoint = previous_state_datapoint.deepcopy()


def destroy():
    global sim, stage, object_prims, datapoint, metrics, constraints, config_cache
    global _loaded_scene_usd_path, _scene_baseline_metrics, _scene_baseline_constraints
    if sim is not None:
        try:
            sim.clear_instance()
        except Exception:
            pass
        sim = None
    stage = None
    object_prims = {}
    datapoint = None
    metrics = {}
    globals()["_logical_robot_metric"] = None
    constraints = {"husky": []}
    config_cache = None
    _loaded_scene_usd_path = None
    _scene_baseline_metrics = None
    _scene_baseline_constraints = {"husky": []}
    simulation_app.close()
