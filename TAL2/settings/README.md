# 

1. Task-agnostic exploration to collect data.

```shell
/root/isaaclab/isaaclab.sh -p src/generate_Aall.py
/root/isaaclab/isaaclab.sh -p src/exploration.py
```
cd /root/gpufree-data/code/tal-vla/TAL2
/root/isaaclab/_isaac_sim/python.sh src/generate_Aall.py
/root/isaaclab/_isaac_sim/python.sh src/exploration.py
适配本机

#cd /root/gpufree-data/code/tal-vla/TAL2
#TAL_ISAAC_SKIP_SIM_STEP=1 TAL_ISAAC_HEADLESS=0 TAL_PERCEPTION_MODE=yolo TAL_YOLO_WARMUP_STEPS=1 TAL_YOLO_CONF=0.35 TAL_EXPLORE_SKIP_PICK=0 TAL_EXPLORE_SKIP_PICKNPLACE=0 TAL_EXPLORE_SKIP_PUSHTO=0 /root/isaaclab/_isaac_sim/python.sh src/exploration.py
进一步适配

cd /root/gpufree-data/code/tal-vla/TAL2

TAL_ISAAC_SKIP_SIM_STEP=1 \
TAL_ISAAC_HEADLESS=0 \
TAL_ISAAC_LOGICAL_ROBOT_MOTION=1 \
TAL_ISAAC_SHOW_LOGICAL_ROBOT_MARKER=1 \
TAL_PERCEPTION_MODE=yolo \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_EXPLORE_SKIP_PICK=0 \
TAL_EXPLORE_SKIP_PICKNPLACE=0 \
TAL_EXPLORE_SKIP_PUSHTO=0 \
/root/isaaclab/_isaac_sim/python.sh src/exploration.py

cd /root/gpufree-data/code/tal-vla/TAL2

TAL_ISAAC_SKIP_SIM_STEP=1 \
TAL_ISAAC_HEADLESS=0 \
TAL_ISAAC_LOGICAL_ROBOT_MOTION=1 \
TAL_ISAAC_SHOW_LOGICAL_ROBOT_MARKER=1 \
TAL_PERCEPTION_MODE=yolo \
TAL_YOLO_CAMERA_PATH=/World/high2 \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_EXPLORE_SKIP_PICK=0 \
TAL_EXPLORE_SKIP_PICKNPLACE=0 \
TAL_EXPLORE_SKIP_PUSHTO=0 \
/root/isaaclab/_isaac_sim/python.sh src/exploration.py

Notes:
- The Isaac Lab environment now loads `/root/Desktop/Collected_exp3/expff.usd`.
- The active scene objects are `Cube`, `SmallPallet`, `BigPallet`, `Bottle2`, `Stool`, `table`, and `Mobie_grasper2`.
- Generated graphs are saved under `data/home/world_expff/`.
- The repo now ships a lightweight local `dgl` compatibility package, so Isaac Lab's
  Python 3.11 / PyTorch 2.7 / CUDA 12.8 environment can run the project without
  installing the legacy external DGL build from `settings/tal.yaml`.
- The reduced symbolic state set is:
  `Outside`, `Inside`, `Up`, `Down`, `Grabbed`, `Free`, `Sticky`, `Non_Sticky`,
  `Fueled`, `Not_Fueled`, `Driven`, `Not_Driven`, `Different_Height`, `Same_Height`.
- The reduced action set is:
  `drop`, `pick`, `moveTo`, `pushTo`, `changeState`, `pickNplaceAonB`.

2. Generate dataset.

```shell
/root/isaaclab/_isaac_sim/python.sh src/generate_and_split_dataset.py
```
cd /root/gpufree-data/code/tal-vla/TAL2
TAL_DATASET_GRAPH_FILE=17.graph \
/root/isaaclab/_isaac_sim/python.sh src/generate_and_split_dataset.py

3. Train action effect feature extractor.

```shell
/root/isaaclab/_isaac_sim/python.sh scripts/train_feature_extractor.py
```
cd /root/gpufree-data/code/tal-vla/TAL2
TAL_AFE_GRAPH_FILE=17.graph \
/root/isaaclab/_isaac_sim/python.sh scripts/train_feature_extractor.py
TAL_AFE_FRESH_START=1 
4. Extract action effect features.

```shell
/root/isaaclab/_isaac_sim/python.sh scripts/generate_action_effect_features.py
```
cd /root/gpufree-data/code/tal-vla/TAL2
TAL_AFE_GRAPH_FILE=17.graph \
/root/isaaclab/_isaac_sim/python.sh scripts/generate_action_effect_features.py

5. Train action proposal (BC)

```shell
/root/isaaclab/_isaac_sim/python.sh scripts/train_action_proposal.py
```
cd /root/gpufree-data/code/tal-vla/TAL2
TAL_APN_GRAPH_FILE=17.graph \
TAL_APN_NUM_WORKERS=0 \
TAL_APN_PIN_MEMORY=0 \
/root/isaaclab/_isaac_sim/python.sh scripts/train_action_proposal.py
#TAL_APN_FRESH_START=1 \
6. Test BC.

```shell
/root/isaaclab/isaaclab.sh -p scripts/test_policy_bc.py --max_samples N
```

7. Test TAL.

```shell
/root/isaaclab/isaaclab.sh -p scripts/test_policy_tal.py
```

8. Test TAL with natural language goal translation via DashScope HTTP API.

```shell
/root/isaaclab/isaaclab.sh scripts/test_policy_tal_nl.py --instruction "Put the milk into the fridge."
```

This entrypoint uses `curl` to call DashScope. Set `DASHSCOPE_API_KEY` or pass `--qwen_api_key`.

8.5 接受自然语言指令，输出下一个动作
export DASHSCOPE_API_KEY=你的key
/root/isaaclab/isaaclab.sh -p scripts/test_next_action_tal_nl.py --instruction "pick up the cube."

9. Train baseline CQL.

```shell
python baseline_train_cql.py
```

10. Train baseline Plan Transformer.

```shell
python baseline_train_pt.py
```





* Since pickup - moveTo - drop can be a minimum sequence length to complete a small task, the length of sequence in training set is set to 1-3.


11.  openpi

cd /root/gpufree-data/code/tal-vla/openpi

PYTHONPATH=. .venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_pro630_lora \
  --policy.dir checkpoints/pi05_pro630_lora/pAndp/14999

###Or
cd /root/gpufree-data/code/tal-vla/openpi
PYTHONPATH=. .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_pro630_lora --policy.dir checkpoints/pi05_pro630_lora/pAndp/14999


###这里要重开个终端

export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850

cd /isaac-sim

TAL_PERCEPTION_MODE=yolo \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller2.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300


export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850
sk-0ad1057b021b4dd3b660a161f1835850

cd /isaac-sim

TAL_PERCEPTION_MODE=yolo \
TAL_ONLINE_HIGH_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_ONLINE_TAL_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAPTURE_MODE=live_camera \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_ONLINE_MOVE_ROBOT_ROOT=0 \
TAL_ONLINE_WARMUP_ROBOT=1 \
TAL_ONLINE_CONTROL_SUBSTEPS=8 \
TAL_ONLINE_DEBUG_CONTROL=1 \
TAL_ONLINE_DEBUG_REPLAN=1 \
TAL_ONLINE_CAMERA_RETRIES=10 \
TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES=120 \
TAL_ONLINE_ALLOW_RESET_FALLBACK=1 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller2.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300


cd /isaac-sim

export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850

TAL_PERCEPTION_MODE=yolo \
TAL_ONLINE_TAL_CAMERA_PATH=/World/high2 \
TAL_YOLO_CAPTURE_MODE=live_camera \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_ONLINE_MOVE_ROBOT_ROOT=0 \
TAL_ONLINE_WARMUP_ROBOT=1 \
TAL_ONLINE_CONTROL_SUBSTEPS=8 \
TAL_ONLINE_DEBUG_CONTROL=1 \
TAL_ONLINE_DEBUG_REPLAN=1 \
TAL_ONLINE_CAMERA_RETRIES=10 \
TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES=120 \
TAL_ONLINE_ALLOW_RESET_FALLBACK=1 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller2.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300


export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850

cd /isaac-sim

TAL_PERCEPTION_MODE=yolo \
TAL_ONLINE_HIGH_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_ONLINE_TAL_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAPTURE_MODE=live_camera \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_ONLINE_MOVE_ROBOT_ROOT=0 \
TAL_ONLINE_WARMUP_ROBOT=1 \
TAL_ONLINE_CONTROL_SUBSTEPS=8 \
TAL_ONLINE_DEBUG_CONTROL=1 \
TAL_ONLINE_DEBUG_REPLAN=1 \
TAL_ONLINE_CAMERA_RETRIES=10 \
TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES=120 \
TAL_ONLINE_ALLOW_RESET_FALLBACK=1 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller2.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300


export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850

cd /isaac-sim

TAL_PERCEPTION_MODE=yolo \
TAL_ONLINE_HIGH_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_ONLINE_TAL_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAPTURE_MODE=live_camera \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_ONLINE_MOVE_ROBOT_ROOT=0 \
TAL_ONLINE_WARMUP_ROBOT=1 \
TAL_ONLINE_CONTROL_SUBSTEPS=8 \
TAL_ONLINE_DEBUG_CONTROL=1 \
TAL_ONLINE_DEBUG_REPLAN=1 \
TAL_ONLINE_CAMERA_RETRIES=10 \
TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES=120 \
TAL_ONLINE_ALLOW_RESET_FALLBACK=1 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller2.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300 \
  2>&1 | tee /tmp/tal_move_debug_0531.log



1.
cd /root/gpufree-data/code/tal-vla/robot_ws
source /opt/ros/jazzy/setup.bash
colcon build

2.
cd /root/gpufree-data/code/tal-vla/robot_ws
bash run_nav2_clean.sh

3.
cd /root/gpufree-data/code/tal-vla/openpi
PYTHONPATH=. .venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_pro630_lora \
  --policy.dir checkpoints/pi05_pro630_lora/pAndp/14999

4.
export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850

cd /isaac-sim

TAL_PERCEPTION_MODE=yolo \
TAL_ONLINE_HIGH_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_ONLINE_TAL_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAPTURE_MODE=live_camera \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_ONLINE_MOVE_ROBOT_ROOT=0 \
TAL_ONLINE_WARMUP_ROBOT=1 \
TAL_ONLINE_CONTROL_SUBSTEPS=8 \
TAL_ONLINE_DEBUG_CONTROL=1 \
TAL_ONLINE_DEBUG_REPLAN=1 \
TAL_ONLINE_CAMERA_RETRIES=10 \
TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES=120 \
TAL_ONLINE_ALLOW_RESET_FALLBACK=1 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller2.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300

export DASHSCOPE_API_KEY=sk-0ad1057b021b4dd3b660a161f1835850

cd /isaac-sim

TAL_PERCEPTION_MODE=yolo \
TAL_YOLO_CAPTURE_MODE=live_camera \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_ONLINE_MOVE_ROBOT_ROOT=0 \
TAL_ONLINE_WARMUP_ROBOT=1 \
TAL_ONLINE_CONTROL_SUBSTEPS=8 \
TAL_ONLINE_DEBUG_CONTROL=1 \
TAL_ONLINE_DEBUG_REPLAN=1 \
TAL_ONLINE_CAMERA_RETRIES=10 \
TAL_ONLINE_CAMERA_PREFLIGHT_RETRIES=120 \
TAL_ONLINE_ALLOW_RESET_FALLBACK=1 \
TAL_ONLINE_WARMUP_DIRECT_SET_JOINTS=1 \
TAL_ONLINE_HIGH_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_ONLINE_TAL_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_YOLO_CAMERA_PATH=/World/Mobie_grasper2/high \
TAL_NAV_ROOT_PRIM_PATH=/World/Mobie_grasper2/firefighter \
TAL_NAV_FLATTEN_ROOT_ATTITUDE=0 \
TAL_NAV_ENFORCE_ROOT_POSE=0 \
TAL_NAV_MAX_TRANSLATION_STEP_M=0.003 \
TAL_NAV_MAX_YAW_STEP_RAD=0.01 \
TAL_NAV_ROOT_UPDATE_INTERVAL_S=0.15 \
TAL_NAV_ACCEPT_FAILED_GOAL_IF_CLOSE=1 \
TAL_NAV_CLOSE_POSITION_TOL_M=0.25 \
TAL_NAV_CLOSE_YAW_TOL_RAD=1.2 \
TAL_ONLINE_DIRECT_SET_JOINTS=1 \
PYTHONPATH=/root/gpufree-data/code/tal-vla:/root/gpufree-data/code/tal-vla/openpi/packages/openpi-client/src \
./python.sh /root/gpufree-data/code/tal-vla/sim_inference_tal_controller4.py \
  --prompt "pick up the cube and put it in pallet" \
  --tal-root /root/gpufree-data/code/tal-vla/TAL2 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --replan-every-n-steps 300 \
  --nav-server-timeout-sec 45 \
  --nav-goal-timeout-sec 120



#git 指令
# 1. 确保进入你的代码目录（这一步最重要，防止报“不是 git 仓库”的错误）
cd ~/gpufree-data/code/tal-vla

# 2. 把刚才上传的视频和所有修改打包
git add .

# 3. 提交并写备注（双引号里的文字可以随便改，比如写你今天做了什么）
git commit -m "26-6-7_1"

# 4. 推送到 GitHub（走你之前配好的免密 SSH，直接回车即可）
git push origin master
