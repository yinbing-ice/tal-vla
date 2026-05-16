"""
generate_Aall.py
distripection: generate Aall for task-agnostic exploration.
"""
import os
import pickle
import random
import colorama
from tqdm import tqdm
from termcolor import cprint
from collections import deque

# from src.envs import husky_ur5 as env
from src.envs import isaac_env as env
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig

colorama.init()


def generate_possible_hl_actions(config, upsample=False, upsample_delta=0.3, shuffle=True):
    possible_hl_actions = []
    p_hl_actions_dict = {}

    # * ---------------------------------------------------------------
    # * Generate possible high-level actions.
    for action in config.possibleActions:
        p_hl_actions_dict[action] = []

        if action == 'moveTo':
            for obj in config.navigation_targets:
                p_hl_actions_dict[action].append({'name': action, 'args': [obj]})

        elif action in ['pick', 'drop']:
            for obj in config.property2Objects['Movable']:
                p_hl_actions_dict[action].append({'name': action, 'args': [obj]})

        elif action in ['pushTo', 'pickNplaceAonB']:
            for obj_a in config.property2Objects['Movable']:
                if action == 'pickNplaceAonB' and obj_a in config.large_objects:
                    continue
                for obj_b in config.place_targets:
                    if obj_a == obj_b:
                        continue
                    p_hl_actions_dict[action].append({'name': action, 'args': [obj_a, obj_b]})

    # * ---------------------------------------------------------------
    # * UpSampling.
    if upsample:
        slightly_up_sample_actions = ['moveTo', 'pick']
        max_num = -1
        for value in p_hl_actions_dict.values():
            if len(value) > max_num:
                max_num = len(value)
        for key, value in p_hl_actions_dict.items():
            delta_num = max_num - len(value)
            # * UpSampling
            if (delta_num > 0):
                delta_num = delta_num if (
                        key not in slightly_up_sample_actions) else delta_num * upsample_delta
                value += random.choices(value, k=int(delta_num))
            possible_hl_actions += value
    else:
        for key, value in p_hl_actions_dict.items():
            possible_hl_actions += value

    possible_hl_actions = deque(possible_hl_actions)
    data_num = len(possible_hl_actions)

    # * ---------------------------------------------------------------
    # * Shuffle
    if shuffle:
        for _ in range(30):
            random.shuffle(possible_hl_actions)
            possible_hl_actions.rotate(random.randint(int(data_num / 5), data_num))

    print('--' * 20)
    print('possible_hl_actions')
    print(possible_hl_actions)
    print('data_num: {}'.format(data_num))
    print('--' * 20)

    return possible_hl_actions


def generate_A_all(config):
    # * Possible high level actions.
    possible_hl_actions = generate_possible_hl_actions(config)  # * without moveTo.
    # * ---------------------------------------------------------------
    # * Initial environment.
    config.world = config.graph_world_name

    # * Select an action.
    env.start(config)
    # 2026-05-10 修改：A_all 只负责验证高层动作是否可执行，不生成训练用状态特征。
    # 因此这里的 saveState 继续保留 Isaac 真值，仅用于 restoreState 回滚仿真；真正的 YOLO 坐标替换在 exploration.py 写 graph 时完成。
    state_id, previous_constraints = env.saveState()
    env.initRootNode()  # * Add 'End' to the datapoint, use 'End' as node final state.
    previous_state_datapoint = env.getDatapoint(config, RESET_DATAPOINT=True)

    A_all = []
    for data_idx in tqdm(range(len(possible_hl_actions))):
        hl_item = possible_hl_actions[data_idx]
        hl_actions = {'actions': [hl_item]}

        # * -----------------------------------------------------------
        # * Execute an action.
        # cprint(hl_actions, 'green')
        try:
            env.execute_collect_data(config, hl_actions, goal_file=None, saveImg=False)
            A_all.append(hl_actions)  # * Execute successful: add to A_all.
            # print(hl_actions)
        except Exception as e:
            # cprint(str(e), 'green')
            if str(e).startswith('Error') or str(e).startswith('Can not complete'):
                pass
            else:
                A_all.append(hl_actions)  # * Execute failure but not Error: add to A_all.

        # * Reload world state.
        env.restoreState(state_id, previous_constraints, previous_state_datapoint)
        env.resetDatapoint(config, previous_state_datapoint)
        # break

    # * ---------------------------------------------------------------
    # * Save graph.
    os.makedirs(os.path.dirname(config.Aall_path), exist_ok=True)
    with open(config.Aall_path, 'wb') as f:
        pickle.dump(A_all, f)
    with open(config.all_possible_actions_path, 'wb') as f:
        pickle.dump(possible_hl_actions, f)

    print('--' * 20)
    cprint('all possible actions num: {}'.format(len(possible_hl_actions)), 'green')
    cprint('A_all actions num: {}'.format(len(A_all)), 'green')
    print('--' * 20)


if __name__ == '__main__':
    args = init_args()
    config = EnvironmentConfig(args)
    # config.display = True

    generate_A_all(config)
