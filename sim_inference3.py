# ./python.sh /root/gpufree-data/code/sim_inference3.py --prompt "pick up the cube"
# uv run scripts/serve_policy.py     --port 8000     policy:checkpoint     --policy.config pi05_pro630_lora     --policy.dir checkpoints/pi05_pro630_lora/sim_room/14999
# 加大小车的阻尼和质量
import argparse
import sys
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser(description="OpenPI Isaac Sim Inference")
parser.add_argument("--prompt", type=str, default="pick up the block", help="The language instruction for the robot")
# 使用 parse_known_args，因为 Isaac Sim 也会传入一些内部参数，我们要避免报错
args, unknown_args = parser.parse_known_args()

# 将未知的参数留给 Isaac Sim (SimulationApp)
sys.argv = [sys.argv[0]] + unknown_args

print(f"--> Current Prompt: {args.prompt}")
# 1. 启动 Isaac Sim (headless=False 表示有界面，可以通过VNC看)
simulation_app = SimulationApp({"headless": False})

import numpy as np
import cv2
import time
from omni.isaac.core import World

import omni.replicator.core as rep

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import DynamicCuboid 
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.materials import PhysicsMaterial

from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import omni.isaac.core.utils.numpy.rotations as rot_utils

# ---------------------------------------------------------
# 配置区域 (请根据你的USD文件修改这里!)
# ---------------------------------------------------------
usd_path = "/root/Desktop/Collected_exp3/expff.usd"
robot_prim_path = "/World/Mobie_grasper2"
cube_prim_path = "/World/Cube"
# plate_path = "/World/plate"
MATERIAL_PRIM_PATH = "/World/PhysicsMaterial" 
CAMERA_HIGH_PATH = os.environ.get("TAL_ONLINE_HIGH_CAMERA_PATH", "/World/Mobie_grasper2/high")
CAMERA_WRIST_PATH = "/World/Mobie_grasper2/firefighter/joint6/wrist"



# 服务端 IP (本机)
SERVER_URL = "http://127.0.0.1:8000"
# ---------------------------------------------------------

def main():
    # 2. 加载场景
    print(f"Loading Stage: {usd_path}")
    from omni.isaac.core.utils.stage import open_stage
    open_stage(usd_path)

    
    # 3. 初始化 World 和 对象
    world = World()


    
    # 初始化机器人
    # 假设你的 USD 里机器人已经配置好了 Articulation Root
    robot = world.scene.add(Articulation(prim_path=robot_prim_path, name="firefighter"))
    # cube = world.scene.add(RigidPrim(prim_path=cube_prim_path, name="target_cube", mass=0.005))
    # plate = world.scene.add(RigidPrim(prim_path=plate_path,name="plate"))

    # try:
    #     target_material = PhysicsMaterial(prim_path=MATERIAL_PRIM_PATH)
    #     cube.apply_physics_material(target_material)
    # except:
    #     pass
    
    # 初始化相机
    cam_high = Camera(prim_path=CAMERA_HIGH_PATH, resolution=(224, 224))
    cam_wrist = Camera(prim_path=CAMERA_WRIST_PATH, resolution=(224, 224))
    cam_high.initialize()
    cam_wrist.initialize()

    # 4. 连接策略服务器
    print("Connecting to Policy Server...")
    # 注意：这里传入 host:port
    policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
    print("Connected!")

    # 5. 开始仿真
    world.reset()

    initial_positions = robot.get_joint_positions()

# 瞬间下发控制指令，把目标位置设为它现在的实际位置！
# 这能让底盘、机械臂和夹爪像被“冻结”一样死死锁在原地，不会乱甩

    # new
    TRAIN_INIT_STATE = np.array([
    -0.12466581, -0.15327631, 1.2, -0.1757595, 1.5070096, -0.320009, 0.13824108], dtype=np.float32)
    JOINT_NAMES_IN_ORDER = [
    "joint1_to_base", 
    "joint2_to_joint1", 
    "joint3_to_joint2", 
    "joint4_to_joint3", 
    "joint5_to_joint4", 
    "joint6_to_joint5", 
    "finger_joint" ]

    sim_dof_names = robot.dof_names
    target_indices = []
    
    for name in JOINT_NAMES_IN_ORDER:
        if name in sim_dof_names:
            target_indices.append(sim_dof_names.index(name))
        else:
            print(f"Warning: 关节 {name} 未在仿真中找到！")
            
    start_positions = robot.get_joint_positions()[target_indices]
    num_steps = 240 
    # 2. 强制设置关节位置 (Teleport)
       # 注意：我们传入 joint_indices，这样就不会影响那些 mimic 关节
    # init_action = ArticulationAction(
    #     joint_positions=TRAIN_INIT_STATE,
    #     joint_indices=np.array(target_indices, dtype=np.int32)
    # )
    
    # 预热几帧
    for i in range(num_steps):
        alpha = (i + 1) / float(num_steps)
        
        # 线性插值公式：当前位置 = 起点 + 比例 * (终点 - 起点)
        interpolated_positions = start_positions + alpha * (TRAIN_INIT_STATE - start_positions)
        
        # 下发插值后的一小步动作
        step_action = ArticulationAction(
            joint_positions=interpolated_positions,
            joint_indices=np.array(target_indices, dtype=np.int32)
        )
        robot.apply_action(step_action)
        world.step()
    final_action = ArticulationAction(
        joint_positions=TRAIN_INIT_STATE,
        joint_indices=np.array(target_indices, dtype=np.int32)
    )
    for _ in range(60):
        robot.apply_action(final_action)
        world.step(render=False)
    print("Starting Inference Loop...")
    
    while simulation_app.is_running():
        world.step()
        
        # --- A. 获取数据 ---
        
        # 1. 获取图像 (RGBA -> RGB)
        # Isaac Sim 相机默认输出 RGBA float32 或 uint8，我们需要 RGB uint8
        img_high_rgba = cam_high.get_rgba()[:, :, :3]
        img_wrist_rgba = cam_wrist.get_rgba()[:, :, :3]
        
        # 确保是 uint8 [0, 255]
        if img_high_rgba.dtype == np.float32:
            img_high_rgb = (img_high_rgba * 255).astype(np.uint8)
            img_wrist_rgb = (img_wrist_rgba * 255).astype(np.uint8)
        else:
            img_high_rgb = img_high_rgba
            img_wrist_rgb = img_wrist_rgba

        img_high = cv2.cvtColor(img_high_rgb, cv2.COLOR_RGB2BGR)
        img_wrist = cv2.cvtColor(img_wrist_rgb, cv2.COLOR_RGB2BGR)
        
        # 2. 获取状态 (7维: 6关节 + 1夹爪)
        # 获取所有关节位置
        # joint_pos = robot.get_joint_positions()
        # current_state = joint_pos[:7].astype(np.float32)

        # new
        all_joint_pos = robot.get_joint_positions()
        all_dof_names = robot.dof_names # 获取当前仿真里的名字顺序
        ordered_state = []
        for name in JOINT_NAMES_IN_ORDER:
            # 找到名字在仿真数组中的索引
            if name in all_dof_names:
                idx = all_dof_names.index(name)
                val = all_joint_pos[idx]
                ordered_state.append(val)
            else:
                print(f"Error: Joint {name} not found in simulation!")
        
        current_state = np.array(ordered_state, dtype=np.float32)

        
        # --- B. 构建 Observation ---
        # Key 必须与 config.py 中 RepackTransform 的输入一致
        obs = {
            "observation/images/cam_high": img_high,
            "observation/images/cam_wrist": img_wrist,
            # Policy 读取的是 data["observation/state"]
            "observation/state": current_state,
            # Policy 读取的是 data["prompt"] (config中映射的是 prompt <- task，所以目标是 prompt)
            "prompt": args.prompt      
        }


        # --- C. 推理 ---
        result = policy.infer(obs)
        actions = result["actions"] # shape: [H, 7]
        
        # 取第一帧动作
        target_action = actions[0]
        
        # --- D. 执行动作 ---
        # 假设模型输出的是 绝对位置 (Absolute Position)
        # 如果模型输出是 Delta，你需要: target = current_state + target_action
        # 但你在 config.py 里配置了 AbsoluteActions output transform，
        # 所以这里收到的应该是 绝对位置，直接应用即可。
        
        # 设置机械臂关节目标 (前6个) 和 夹爪 (第7个)
        # 注意：你需要确认 robot.apply_action 是按什么顺序接受参数的
        
        # 创建全量的 action 数组 (如果 sim 里关节多于 7 个)
        full_action = np.zeros_like(all_joint_pos)
        full_action[:7] = target_action
        # print(f"Delta        : {target_action[:6] - current_state[:6]}")
        # from omni.isaac.core.utils.types import ArticulationAction
        # 发送位置控制指令
        # action_cmd = ArticulationAction(joint_positions=full_action)
        # robot.apply_action(action_cmd)
        action_cmd = ArticulationAction(
            joint_positions=target_action,
            joint_indices=np.array(target_indices, dtype=np.int32)
        )
        
        robot.apply_action(action_cmd) 

        
    simulation_app.close()

if __name__ == "__main__":
    main()
