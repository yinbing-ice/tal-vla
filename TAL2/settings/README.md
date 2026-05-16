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
#TAL_ISAAC_SKIP_SIM_STEP=1 \
TAL_ISAAC_HEADLESS=1 \
TAL_PERCEPTION_MODE=yolo \
TAL_YOLO_WARMUP_STEPS=1 \
TAL_YOLO_CONF=0.35 \
TAL_EXPLORE_SKIP_PICK=0 \
TAL_EXPLORE_SKIP_PICKNPLACE=0 \
TAL_EXPLORE_SKIP_PUSHTO=0 \
/root/isaaclab/_isaac_sim/python.sh src/exploration.py

TAL_ISAAC_SKIP_SIM_STEP=1 TAL_ISAAC_HEADLESS=1 TAL_PERCEPTION_MODE=yolo TAL_YOLO_WARMUP_STEPS=1 TAL_YOLO_CONF=0.35 TAL_EXPLORE_SKIP_PICK=0 TAL_EXPLORE_SKIP_PICKNPLACE=0 TAL_EXPLORE_SKIP_PUSHTO=0 /root/isaaclab/_isaac_sim/python.sh src/exploration.py
进一步适配


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

3. Train action effect feature extractor.

```shell
/root/isaaclab/_isaac_sim/python.sh scripts/train_feature_extractor.py
```
cd /root/gpufree-data/code/tal-vla/TAL2

TAL_AFE_GRAPH_FILE=11.graph \
/root/isaaclab/_isaac_sim/python.sh scripts/train_feature_extractor.py
TAL_AFE_FRESH_START=1 
4. Extract action effect features.

```shell
/root/isaaclab/_isaac_sim/python.sh scripts/generate_action_effect_features.py
```
cd /root/gpufree-data/code/tal-vla/TAL2

TAL_AFE_GRAPH_FILE=11.graph \
/root/isaaclab/_isaac_sim/python.sh scripts/generate_action_effect_features.py

5. Train action proposal (BC)

```shell
/root/isaaclab/_isaac_sim/python.sh scripts/train_action_proposal.py
```
cd /root/gpufree-data/code/tal-vla/TAL2

TAL_APN_GRAPH_FILE=11.graph \

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