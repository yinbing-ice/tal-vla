# sim_inference_tal_controller4.py 说明

本文档说明 `/root/gpufree-data/code/tal-vla/sim_inference_tal_controller4.py` 的作用、主流程和关键函数调用链。

## 一句话概括

`sim_inference_tal_controller4.py` 是一个 Isaac Sim 在线闭环控制脚本，用来把 TAL 高层无任务规划、YOLO 场景感知、Nav2 导航和 OpenPI/VLA 低层抓取控制串起来。

目标流程大致是：

```text
用户自然语言任务
  -> TAL + YOLO 生成当前 scene graph
  -> TAL 规划下一步子任务
  -> 如果是 moveTo，则调用 Nav2，让小车平动接近目标
  -> 如果不是导航子任务，则调用 OpenPI，根据相机和关节状态输出机械臂动作
  -> 在 Isaac Sim 中执行动作
```

当前主要用于实验 `move -> pick -> place` 这条链路。

## 输入与环境变量

脚本通过命令行参数接收任务、TAL 根目录、OpenPI server 地址和导航超时等信息。

主要参数：

- `--prompt`：用户任务，例如 `pick up the cube and put it in pallet`
- `--tal-root`：TAL2 目录，通常是 `/root/gpufree-data/code/tal-vla/TAL2`
- `--server-host` / `--server-port`：OpenPI policy server 地址
- `--replan-every-n-steps`：每隔多少低层控制步重新调用 TAL
- `--nav-server-timeout-sec`：等待 Nav2 action server 的超时时间
- `--nav-goal-timeout-sec`：单次导航目标的等待时间

主要环境变量：

- `TAL_ONLINE_HIGH_CAMERA_PATH`：OpenPI/VLA 使用的第一人称 high 相机，当前默认 `/World/Mobie_grasper2/high`
- `TAL_ONLINE_TAL_CAMERA_PATH`：TAL/YOLO 可单独使用的相机
- `TAL_YOLO_CAMERA_PATH`：YOLO 反投影使用的相机 prim path
- `TAL_YOLO_CAPTURE_MODE`：YOLO 图像来源，常用 `live_camera`
- `TAL_YOLO_CONF`：YOLO 检测置信度阈值
- `TAL_ONLINE_WARMUP_ROBOT`：是否启动时把机械臂 warmup 到初始姿态
- `TAL_ONLINE_DIRECT_SET_JOINTS`：OpenPI 控制时是否直接设置关节位置
- `TAL_NAV_FLATTEN_ROOT_ATTITUDE`：导航阶段是否压平 root 的 roll/pitch
- `TAL_NAV_ENFORCE_ROOT_POSE`：是否在仿真 step 后强制写回 root pose
- `TAL_NAV_MAX_TRANSLATION_STEP_M`：每次 root 平移最大步长
- `TAL_NAV_MAX_YAW_STEP_RAD`：每次 yaw 更新最大步长
- `TAL_NAV_ACCEPT_FAILED_GOAL_IF_CLOSE`：Nav2 失败但距离足够近时是否接受导航结果

注意：controller4 当前直接使用 `RobotRootPoseController(robot)`，没有真正读取 `TAL_NAV_ROOT_PRIM_PATH` 来创建 XFormPrim root controller。

## 主要模块分工

### 1. 参数、相机和机器人常量

代码开头定义命令行参数和路径常量。

关键变量：

- `CAMERA_HIGH_PATH`
- `CAMERA_WRIST_PATH`
- `CAMERA_TAL_PATH`
- `ROBOT_START_WORLD_POSITION`
- `TRAIN_INIT_STATE`
- `JOINT_NAMES_IN_ORDER`
- `NAV_MAP_YAML_PATH`

`CAMERA_HIGH_PATH` 是给 OpenPI 的第一人称图像，`CAMERA_WRIST_PATH` 是腕部相机，`CAMERA_TAL_PATH` 是 TAL/YOLO 规划用图像。

### 2. Isaac 初始化与保护工具

相关函数：

- `_initialize_robot_handles_if_needed(...)`
- `_restore_robot_pose_if_available(...)`
- `_capture_current_robot_pose(...)`
- `_step_world_with_root_guard(...)`
- `_render_world_if_possible(...)`
- `warm_up_robot(...)`
- `warm_up_cameras(...)`
- `wait_for_camera_frames(...)`

这些函数负责：

- 等待 Isaac articulation 句柄可用
- 在 reset fallback 前后尽量保留机械臂姿态
- 预热相机，确保 `cam_high`、`cam_wrist`、`cam_tal` 都能拿到图像
- warmup 机械臂到训练初始状态
- 在推进 `world.step()` 后可选地用 root guard 固定机器人 root

### 3. TAL 运行时初始化

核心类和函数：

- `TALControllerConfig`
- `TALRuntimeContext`
- `initialize_tal_runtime(...)`
- `_build_env_config(...)`
- `_load_required_model(...)`

`initialize_tal_runtime(...)` 做的事最多：

1. 把 `TAL2` 加进 `sys.path`
2. 动态 import TAL2 里的配置、规划、训练和 scene graph 模块
3. 构建两个配置：
   - `sim_env_config`：给 Isaac/TAL 当前观测用
   - `planner_env_config`：给符号规划和模型推理用
4. 加载 AFE 和 APN checkpoint
5. 加载 `action_effect_features_avg.pkl`
6. 初始化 TAL 的 `approx` 后端
7. 返回 `TALRuntimeContext`

### 4. YOLO + scene graph 链路

controller4 本身不直接 import `ultralytics.YOLO`，它通过 TAL2 的 `getObservedDatapoint(...)` 间接调用 YOLO。

关键调用链：

```text
capture_rgb_images(...)
  -> images["cam_tal"]
  -> TALSceneGraphProvider.get_current_scene_graph(...)
  -> TALSceneGraphProvider._refresh_live_datapoint(...)
  -> isaac_env.getObservedDatapoint(..., image_rgb=image_rgb)
  -> TAL2/src/envs/perception_state.py
  -> apply_yolo_observation_to_datapoint(...)
  -> detector.detect(image_rgb=image_rgb)
```

相关类：

- `TALSceneGraphProvider`

相关函数：

- `capture_rgb_images(...)`
- `TALSceneGraphProvider.get_current_scene_graph(...)`

YOLO 主要用于替换物体坐标。当前 TAL2 里的逻辑是：

- `husky` 小车坐标保留 Isaac 真值/自定位
- `table` 坐标保留真值/标定
- `cube`、`bottle`、`smallpallet`、`bigpallet`、`stool` 等物体用 YOLO 估计位置

### 5. TAL 高层规划

核心类：

- `LazyTALPlanner`
- `TALPlanResult`
- `ParsedTALSubtask`

核心函数：

- `LazyTALPlanner.plan_first_action(...)`
- `parse_tal_subtask(...)`
- `derive_executable_subtask(...)`
- `build_fused_prompt(...)`
- `format_tal_action(...)`

调用流程：

```text
当前 scene graph
  -> scene_graph_json_to_dgl(...)
  -> plan_with_natural_language_instruction(...)
  -> predicted_actions
  -> 取第一个 action
  -> parse_tal_subtask(...)
  -> derive_executable_subtask(...)
```

`derive_executable_subtask(...)` 会把某些组合动作转成导航动作。例如：

```text
pickNplaceAonB(cube, pallet)
  -> moveTo(cube)
```

也就是说当前实现中，TAL 可能先规划出 pick/place 类动作，但 controller4 会先派生出一个 `moveTo(...)`，让小车先靠近目标。

### 6. 导航目标构造

相关数据结构：

- `NavigationGoal`
- `NavOccupancyMap`

关键函数：

- `build_navigation_goal(...)`
- `infer_navigation_approach_distance(...)`
- `infer_navigation_approach_direction(...)`
- `project_nav_goal_to_free_space(...)`
- `navigation_goal_is_close_enough(...)`

`build_navigation_goal(...)` 的作用是根据 TAL 目标物体算出小车应该停的位置。

当前逻辑：

```text
读取 isaac_env.metrics
  -> 找目标物体坐标 target_xy
  -> 找小车坐标 robot_xy
  -> 按 approach distance 算一个离目标有距离的 goal_xy
  -> 算朝向目标的 yaw
  -> 返回 NavigationGoal
```

注意：当前 controller4 里 `build_navigation_goal(...)` 用的是 `isaac_env.metrics`，不是刚生成的 YOLO scene graph 坐标。也就是说 YOLO 用于 TAL 当前图，导航目标这里仍主要依赖 Isaac metrics。

当前版本还把 occupancy map 投影逻辑注释掉了，直接使用自己算出的 `goal_xy`。

### 7. IsaacNavBridge：把 Nav2 输出转成 Isaac root 平动

核心类：

- `RobotRootPoseController`
- `RobotRootPoseGuard`
- `IsaacNavBridge`
- `SubprocessNav2GoalClient`
- `PendingNavigation`

`IsaacNavBridge` 是当前 move 功能的核心。

它做三件事：

1. 启动外部 ROS 子进程 `isaac_nav_bridge_runner.py`
2. 持续向 ROS 侧发布 `/clock`、`/odom`、`/scan`、`/tf` 所需状态
3. 接收 Nav2 输出的 `/cmd_vel`，然后在 Isaac 里积分 `_x/_y/_yaw`，通过 `set_world_pose(...)` 平移小车 root

核心方法：

- `_start_bridge_process()`：启动 ROS 桥接子进程
- `_publish_state()`：把当前 x/y/yaw 等状态通过 UDP 发给 ROS 侧
- `_poll_cmd()`：从 ROS 侧读取 Nav2 `/cmd_vel`
- `advance(dt)`：每个仿真步更新导航状态
- `sync_from_current_root()`：导航开始前同步当前 Isaac root
- `_apply_root_pose()`：把内部状态写回 Isaac 机器人 root
- `set_active_goal(...)`：设置当前是否处于导航阶段
- `settle_to_goal_pose(...)`：导航结束后强行落到目标位姿，当前主流程里已注释掉不用

`SubprocessNav2GoalClient` 则负责启动 `nav2_goal_runner.py`，向 Nav2 发送单次 `NavigateToPose` action goal，并在子线程里等待结果。

### 8. OpenPI/VLA 低层控制

核心函数：

- `infer_action(...)`
- `apply_robot_action(...)`
- `read_robot_state(...)`
- `capture_rgb_images(...)`

OpenPI 输入格式：

```python
obs = {
    "observation/images/cam_high": images["cam_high"],
    "observation/images/cam_wrist": images["cam_wrist"],
    "observation/state": state,
    "prompt": fused_prompt,
}
```

`infer_action(...)` 调用 OpenPI policy server：

```python
result = policy_client.infer(obs)
return result["actions"][0]
```

`apply_robot_action(...)` 把 OpenPI 输出的 7 维动作应用到机械臂关节：

- 优先 `set_joint_positions`，如果设置了 `TAL_ONLINE_DIRECT_SET_JOINTS=1`
- 否则尝试 `set_joint_position_targets`
- 再否则回退 `robot.apply_action(...)`

每次动作会重复执行多个 substep，由 `TAL_ONLINE_CONTROL_SUBSTEPS` 控制。

## main() 主流程

`main()` 是整个脚本的入口。

### 阶段 1：初始化 TAL 和 Isaac

调用：

```text
initialize_tal_runtime(...)
World(...)
Articulation(...)
Camera(...)
TALSceneGraphProvider(...)
LazyTALPlanner(...)
```

创建：

- Isaac world
- robot articulation
- high 相机
- wrist 相机
- 可选 TAL/YOLO 独立相机
- TAL planner
- scene graph provider

### 阶段 2：机器人句柄和姿态准备

调用：

```text
_capture_current_robot_pose(...)
RobotRootPoseController(robot)
RobotRootPoseGuard(...)
_initialize_robot_handles_if_needed(...)
_restore_robot_pose_if_available(...)
warm_up_robot(...)
```

当前 controller4 明确写了：

```python
robot_root_controller = RobotRootPoseController(robot)
```

也就是说 root controller 直接控制 `robot` articulation，而不是用 `TAL_NAV_ROOT_PRIM_PATH` 创建 `XFormPrim`。

### 阶段 3：启动导航桥接

调用：

```text
IsaacNavBridge(...)
SubprocessNav2GoalClient(...)
```

`IsaacNavBridge` 会启动 `isaac_nav_bridge_runner.py`，后者在 ROS 侧发布 odom/clock/tf/scan，并接收 Nav2 的 cmd_vel。

### 阶段 4：相机预热

调用：

```text
wait_for_camera_frames(...)
warm_up_cameras(...)
```

确保高位相机、腕部相机、TAL 相机都有可用图像。

### 阶段 5：在线闭环主循环

每轮循环大致是：

```text
advance_simulation(...)
capture_rgb_images(...)
read_robot_state(...)
如果需要 replan:
    scene_graph_provider.get_current_scene_graph(...)
    tal_planner.plan_first_action(...)
    parse_tal_subtask(...)
如果当前子任务是 moveTo:
    build_navigation_goal(...)
    nav_bridge.sync_from_current_root()
    nav_bridge.set_active_goal(nav_goal)
    nav_client.send_goal(...)
    while Nav2 未返回:
        advance_simulation(...)
    结束导航，强制下一轮 TAL 重新规划
否则:
    连接 OpenPI
    infer_action(...)
    apply_robot_action(...)
```

## 当前 controller4 的实验性行为

这份 controller4 不是一个完全干净的稳定版，里面有一些明显的调试/实验痕迹，需要特别注意。

### 1. `TAL_NAV_ROOT_PRIM_PATH` 当前没有真正生效

虽然运行命令里可能设置了：

```bash
TAL_NAV_ROOT_PRIM_PATH=/World/Mobie_grasper2/firefighter
```

但 controller4 当前 main 里直接写的是：

```python
robot_root_controller = RobotRootPoseController(robot)
```

因此它没有像 controller2 那样解析 `TAL_NAV_ROOT_PRIM_PATH` 再创建 `XFormPrim`。

### 2. 导航结束后没有调用 `settle_to_goal_pose(...)`

当前导航成功后，旧逻辑被注释掉：

```python
# nav_bridge.settle_to_goal_pose(nav_goal)
```

现在实际执行的是：

```python
nav_bridge.set_active_goal(None)
nav_bridge._idle_root_hold_enabled = False
force_replan = True
```

这意味着导航结束后不会强行把小车瞬移到目标点，而是释放底盘并强制下一轮 TAL 重新规划。

### 3. 导航期间是否停止主要依赖 Nav2 action 返回

主循环在导航阶段会卡在：

```python
while not pending_nav.done_event.wait(timeout=0.0):
    advance_simulation(...)
```

只有 Nav2 子进程返回结果后，才会进入后续 TAL/OpenPI 流程。当前没有在 while 中主动检查“机器人已经靠近目标就本地停止”的逻辑。

### 4. YOLO 用于 scene graph，但导航目标仍读取 `isaac_env.metrics`

TAL 当前 scene graph 会走 `getObservedDatapoint(..., image_rgb=...)`，也就是会使用 YOLO 观测物体坐标。

但 `build_navigation_goal(...)` 里读取的是：

```python
metrics = getattr(isaac_env, "metrics", None)
```

因此导航目标点计算仍主要来自 Isaac metrics，而不是刚刚生成的 YOLO scene graph。

## 最重要的调用链总结

### TAL + YOLO

```text
main loop
  -> capture_rgb_images
  -> TALSceneGraphProvider.get_current_scene_graph
  -> isaac_env.getObservedDatapoint
  -> TAL2/src/envs/perception_state.py
  -> YOLO 检测物体坐标
  -> scene_graph_translator.datapoint_to_scene_graph_json
```

### TAL 规划

```text
LazyTALPlanner.plan_first_action
  -> scene_graph_json_to_dgl
  -> plan_with_natural_language_instruction
  -> predicted_actions
  -> parse_tal_subtask
  -> derive_executable_subtask
```

### moveTo 导航

```text
build_navigation_goal
  -> nav_bridge.sync_from_current_root
  -> nav_bridge.set_active_goal
  -> SubprocessNav2GoalClient.send_goal
  -> nav2_goal_runner.py
  -> Nav2 NavigateToPose
  -> isaac_nav_bridge_runner.py 接收 /cmd_vel
  -> IsaacNavBridge.advance
  -> RobotRootPoseController.set_world_pose
```

### OpenPI 控制

```text
capture_rgb_images
  -> read_robot_state
  -> infer_action
  -> WebsocketClientPolicy.infer
  -> apply_robot_action
  -> robot.set_joint_positions / set_joint_position_targets / apply_action
  -> advance_simulation
```

## 文件外部依赖

controller4 主要依赖：

- `TAL2/src/envs/isaac_env.py`
- `TAL2/src/envs/perception_state.py`
- `TAL2/src/tal/utils_planning.py`
- `TAL2/src/tal/scene_graph_translator.py`
- `TAL2/src/tal/utils_training.py`
- `isaac_nav_bridge_runner.py`
- `nav2_goal_runner.py`
- `openpi_client.websocket_client_policy.WebsocketClientPolicy`

## 当前最值得继续检查的点

1. 是否要在导航 while 中加入“本地 close-stop”，避免 Nav2 不返回时无法进入 OpenPI。
2. `build_navigation_goal(...)` 是否要使用 YOLO scene graph 坐标，而不是 Isaac metrics。
3. root controller 是否应该继续用 `RobotRootPoseController(robot)`，还是恢复成明确的底盘 root `XFormPrim`。
4. 导航结束后是否应该保持底盘 root、释放底盘，还是根据阶段选择不同策略。
5. controller4 中大量注释掉的导航完成逻辑是否需要整理成清晰开关，避免后续调参时混乱。
