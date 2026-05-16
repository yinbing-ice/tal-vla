import os
import signal
import time
import random
import pickle
import torch
import colorama
import numpy as np
import networkx as nx
from copy import deepcopy
from termcolor import cprint
from collections import deque
from torch.utils.data import WeightedRandomSampler

from src.config.config import init_args
from src.envs import isaac_env as env
# from src.envs import husky_ur5 as env
from src.envs.CONSTANTS import EnvironmentConfig
from src.envs.perception_state import apply_yolo_observation_to_datapoint

colorama.init()


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _explore_debug(message):
    # 2026-05-11 修改：默认关闭 exploration 调试日志，避免 YOLO 探索时终端刷屏。
    # 需要排查卡点时设置 TAL_EXPLORE_DEBUG=1 即可恢复这些细节输出。
    if _env_flag("TAL_EXPLORE_DEBUG", False):
        print(f"[ExploreDebug] {message}", flush=True)


def _show_action_log():
    # 2026-05-11 修改：动作序列是 exploration 运行状态的核心进度，默认保留显示；
    # 如需完全安静运行，可设置 TAL_EXPLORE_SHOW_ACTION=0。
    return _env_flag("TAL_EXPLORE_SHOW_ACTION", True)


class ActionExecutionTimeout(TimeoutError):
    pass


def _raise_action_timeout(signum, frame):
    del signum, frame
    raise ActionExecutionTimeout("Timed out while executing high-level action.")


def _execute_action_with_timeout(config, hl_actions):
    # 2026-05-11 修改：YOLO 感知接入后，个别 Isaac 高层动作可能在 execute_collect_data 内卡住。
    # 这里给单个动作加超时保护，超时后跳过该动作并回滚，避免整次 exploration 无响应。
    if os.environ.get("TAL_ISAAC_HEADLESS", "1").lower() in {"0", "false", "no", "off"}:
        # 2026-05-11 修改：GUI 模式下 SIGALRM 可能打断 Isaac viewport 绘制回调，导致 traceback。
        # 调试界面时不使用信号超时，配合动作池过滤避免抽到容易卡住的复杂动作。
        return env.execute_collect_data(config, hl_actions, goal_file=None, saveImg=False)

    timeout_s = int(os.environ.get("TAL_EXPLORE_ACTION_TIMEOUT_S", "8"))
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        return env.execute_collect_data(config, hl_actions, goal_file=None, saveImg=False)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_action_timeout)
    signal.alarm(timeout_s)
    try:
        return env.execute_collect_data(config, hl_actions, goal_file=None, saveImg=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _restore_with_timeout(config, state_id, previous_constraints, previous_state_datapoint):
    # 2026-05-11 修改：动作超时后 Isaac/PhysX 可能处于不稳定状态，restoreState 也可能卡住。
    # 给回滚过程加超时；如果回滚失败，跳过当前分支，避免 env.destroy() 关闭 SimulationApp。
    timeout_s = int(os.environ.get("TAL_EXPLORE_RESTORE_TIMEOUT_S", "8"))
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        env.restoreState(state_id, previous_constraints, previous_state_datapoint)
        env.resetDatapoint(config, previous_state_datapoint)
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_action_timeout)
    signal.alarm(timeout_s)
    try:
        env.restoreState(state_id, previous_constraints, previous_state_datapoint)
        env.resetDatapoint(config, previous_state_datapoint)
    except Exception as exc:
        # 2026-05-11 修改：不要在 exploration 过程中调用 env.destroy()，否则会直接关闭 Isaac App 并退出。
        _explore_debug(f'restore timeout/failure: {exc}; skip current branch')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_Aall_and_split(config, upsample=True):
    with open(config.Aall_path, 'rb') as f:
        Aall = pickle.load(f)
    possible_action_set = []  # * without moveTo
    moveTo_set = []  # * moveTo
    skip_picknplace = os.environ.get("TAL_EXPLORE_SKIP_PICKNPLACE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    skip_pushto = os.environ.get("TAL_EXPLORE_SKIP_PUSHTO", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    skip_pick = os.environ.get("TAL_EXPLORE_SKIP_PICK", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    for action in Aall:
        # 2026-05-11 修改：旧 A_all 里可能包含初始状态不可执行的 standalone drop。
        # drop 需要夹爪已有物体，探索过程会在 pick 后自动插入 drop，因此这里从随机动作池过滤掉。
        if action['actions'][0]['name'] == 'drop':
            continue
        if skip_picknplace and action['actions'][0]['name'] == 'pickNplaceAonB':
            # 2026-05-11 修改：pickNplaceAonB 是组合动作，当前 Isaac GUI/YOLO 调试阶段容易长时间卡住。
            # 默认先从随机探索动作池过滤；如需恢复，设置 TAL_EXPLORE_SKIP_PICKNPLACE=0。
            continue
        if skip_pushto and action['actions'][0]['name'] == 'pushTo':
            # 2026-05-11 修改：pushTo 在当前 Isaac 简化后端中也可能长时间卡住。
            # 默认先过滤，优先验证 YOLO 感知 graph 生成链路；如需恢复，设置 TAL_EXPLORE_SKIP_PUSHTO=0。
            continue
        if skip_pick and action['actions'][0]['name'] == 'pick':
            # 2026-05-11 修改：当前 Isaac 后端中 pick 动作也会超时。
            # 默认先过滤，只保留 moveTo 验证 YOLO graph 生成链路；如需恢复，设置 TAL_EXPLORE_SKIP_PICK=0。
            continue
        if action['actions'][0]['name'] == 'moveTo':
            moveTo_set.append(action)
        else:
            possible_action_set.append(action)

    if len(possible_action_set) == 0 and len(moveTo_set) > 0:
        # 2026-05-11 修改：调试 YOLO graph 链路时可能过滤掉 pick/pushTo/pickNplace，
        # 此时只剩 moveTo；原逻辑不会从 moveTo_set 采样 root 动作，这里将 moveTo 作为兜底动作池。
        _explore_debug('possible_action_set is empty; fallback to moveTo_set')
        possible_action_set = list(moveTo_set)

    print('--' * 20)
    cprint('Aall actions num: {}'.format(len(Aall)), 'green')  # * 1364
    cprint('possible_action_set num: {}'.format(len(possible_action_set)), 'green')  # * 1332
    cprint('moveTo_set num: {}'.format(len(moveTo_set)), 'green')  # * 32
    print('--' * 20)

    # * ---------------------------------------------------------------
    # * UpSampling.
    if upsample:
        hl_actions_upsample = []
        hl_actions_dict = {}
        for action in possible_action_set:
            action_name = action['actions'][0]['name']
            if action_name not in hl_actions_dict:
                hl_actions_dict[action_name] = [action]
            else:
                hl_actions_dict[action_name].append(action)

        max_num = -1
        for value in hl_actions_dict.values():
            if len(value) > max_num:
                max_num = len(value)
        for key, value in hl_actions_dict.items():
            delta_num = max_num - len(value)
            if key == 'pick':
                delta_num = delta_num / 3
            # * UpSampling
            if (delta_num > 0):
                value += random.choices(value, k=int(delta_num))
            hl_actions_upsample += value

        possible_action_set = deque(hl_actions_upsample)
        data_num = len(possible_action_set)

        # * -----------------------------------------------------------
        # * Shuffle
        for _ in range(10):
            random.shuffle(possible_action_set)
            possible_action_set.rotate(random.randint(int(data_num / 5), data_num))

        print('--' * 20)
        cprint('Up-sample', 'green')
        cprint('possible_action_set num: {}'.format(len(possible_action_set)), 'green')  # * 4239
        cprint('moveTo_set num: {}'.format(len(moveTo_set)), 'green')  # * 32
        print('--' * 20)

    return possible_action_set, moveTo_set


def select_actions_from_set(action_set, actions_selected_num, child_actions):
    """Random select high level actions."""
    if len(action_set) == 0:
        # 2026-05-11 修改：动作过滤后如果动作池为空，给出明确错误，避免 WeightedRandomSampler 报晦涩空张量错误。
        raise RuntimeError("No candidate actions left after exploration filters.")
    # * Convert actions_selected_num to sample_weights.
    # * actions_selected_num: record how many times each action was selected.
    # * sample_weights: the more times selected in the history, the lower the probability.
    sample_weights = 1 / actions_selected_num  # 1 / 0 -> inf.
    sample_weights[torch.isinf(sample_weights)] = 0  # * inf -> 0.
    while True:
        # data = random.sample(possible_hl_actions, 1)
        # * data: list with num_samples elems.
        data = list(WeightedRandomSampler(weights=sample_weights, num_samples=1))
        data_idx = data[0]
        if action_set[data_idx] in child_actions:
            continue
        else:
            hl_item = action_set[data_idx]['actions'][0]
            hl_actions = action_set[data_idx]
            # * Add selected num. If this action raise error, it will be set to 0 later.
            actions_selected_num[data_idx] += 1
            return hl_item, hl_actions, data_idx, actions_selected_num


def build_training_datapoint(config, true_metrics, true_constraints, return_raw=False):
    # 2026-05-10 修改：训练 graph 中保存 YOLO 观测坐标，而不是直接保存 Isaac 上帝视角物体坐标。
    # true_metrics 仍作为 world_state 保存给 restoreState 使用，保证探索回滚不受视觉噪声影响。
    _explore_debug('build_training_datapoint: get raw datapoint')
    raw_datapoint = env.getDatapoint(config, RESET_DATAPOINT=True)
    _explore_debug('build_training_datapoint: apply perception observation')
    observed_datapoint = apply_yolo_observation_to_datapoint(
        config,
        raw_datapoint,
        true_metrics=true_metrics,
        constraints=true_constraints,
    )
    _explore_debug('build_training_datapoint: done')
    if return_raw:
        # 2026-05-11 修改：同时返回仿真真值 datapoint，后续 restoreState/resetDatapoint 只使用它回滚环境；
        # graph 节点的 state 仍保存 YOLO 观测版，用于后续训练。
        return observed_datapoint, raw_datapoint
    return observed_datapoint


def explore_from_node_i(config, graph, start_node_id, graph_last_node_id, explore_step,
                        possible_hl_actions,
                        actions_selected_num):
    (possible_action_set, moveTo_set) = possible_hl_actions
    # * Reload world state of start_node_id.
    child_actions = graph.nodes[start_node_id]['child_actions']
    state_id = graph.nodes[start_node_id]['world_state']
    previous_constraints = deepcopy(graph.nodes[start_node_id]['pre_constraints'])
    # 2026-05-11 修改：state 是训练用 YOLO 观测状态；rollback_state 才是仿真回滚用真值状态。
    previous_state_datapoint = deepcopy(
        graph.nodes[start_node_id].get('rollback_state', graph.nodes[start_node_id]['state'])
    )
    if start_node_id == 0 and graph_last_node_id == 0:
        # 2026-05-11 修改：第一次从 root 节点探索时，环境刚由 env.start() 初始化到 root 状态，
        # 这里重复 restoreState 会在 Isaac/Replicator 参与后偶发卡住，因此跳过这次不必要的回滚。
        _explore_debug('restore node 0: skipped initial root restore')
    else:
        _explore_debug(f'restore node {start_node_id}: start')
        _restore_with_timeout(config, state_id, previous_constraints, previous_state_datapoint)
        _explore_debug(f'restore node {start_node_id}: restoreState done')
        _explore_debug(f'restore node {start_node_id}: resetDatapoint done')

    cursor_start = start_node_id
    cursor_end = graph_last_node_id + 1

    # * ---------------------------------------------------------------
    # * Process action 'pick'.
    PRE_PICK = None
    PRE_PICK_AND_MOVE = False
    held_obj = env.get_held_object(previous_constraints)
    if held_obj is not None:
        PRE_PICK = {'name': 'pick', 'args': [held_obj]}
    # * ---------------------------------------------------------------
    step = 0
    data_idx = None
    PRE_UP = None
    error_act_num = 0
    while (step < explore_step):
        _explore_debug(f'explore loop: step={step}/{explore_step}, cursor_end={cursor_end}')
        if error_act_num > 30:
            break
        # time.sleep(1)

        # * -----------------------------------------------------------
        # * Select actions.
        # * Last exploration step.
        if (step + 1) == explore_step:
            # * Process actions: climbUp/climbDown and moveUp/moveDown.
            if (PRE_UP is not None):
                if (PRE_UP['name'] == 'climbUp'):
                    PRE_UP['name'] = 'climbDown'
                elif (PRE_UP['name'] == 'moveUp'):
                    PRE_UP['name'] = 'moveDown'
                hl_item = deepcopy(PRE_UP)
                hl_actions = {'actions': [hl_item]}
                data_idx = None
                PRE_UP = None
            # * Process actions: pick and drop.
            elif (PRE_PICK is not None):
                PRE_PICK['name'] = 'drop'
                hl_item = deepcopy(PRE_PICK)
                hl_actions = {'actions': [hl_item]}
                data_idx = None
                PRE_PICK = None
            # * Process other actions, random select.
            else:
                _explore_debug('select action from possible_action_set')
                hl_item, hl_actions, data_idx, actions_selected_num = select_actions_from_set(
                    possible_action_set, actions_selected_num, child_actions
                )
        else:  # * Random select high level actions.
            rnd_val = np.random.rand()
            # * Process actions: climbUp/climbDown and moveUp/moveDown.
            if (rnd_val > 0.7) and (PRE_UP is not None):
                if (PRE_UP['name'] == 'climbUp'):
                    PRE_UP['name'] = 'climbDown'
                elif (PRE_UP['name'] == 'moveUp'):
                    PRE_UP['name'] = 'moveDown'
                hl_item = deepcopy(PRE_UP)
                hl_actions = {'actions': [hl_item]}
                data_idx = None
                PRE_UP = None
            # * Process actions: pick and drop.
            elif PRE_PICK is not None:
                # * pick --> moveTo --> drop
                if PRE_PICK_AND_MOVE is False:
                    hl_actions = random.choice(moveTo_set)
                    hl_item = hl_actions['actions'][0]
                    data_idx = None
                    PRE_PICK_AND_MOVE = True
                else:
                    PRE_PICK['name'] = 'drop'
                    hl_item = deepcopy(PRE_PICK)
                    hl_actions = {'actions': [hl_item]}
                    data_idx = None
                    PRE_PICK = None
                    PRE_PICK_AND_MOVE = False
            # * Process other actions, random select.
            else:
                _explore_debug('select action from possible_action_set')
                hl_item, hl_actions, data_idx, actions_selected_num = select_actions_from_set(
                    possible_action_set, actions_selected_num, child_actions
                )

        # * -----------------------------------------------------------
        # * Execute action.
        if _show_action_log():
            cprint('Action num: {} '.format(cursor_end), color='green', end='')
            cprint(hl_actions, 'green')
        try:
            _explore_debug('execute action: start')
            done = _execute_action_with_timeout(config, hl_actions)
            _explore_debug('execute action: done')
            error_act_num = 0
            if hl_item['name'] in ['climbDown', 'moveDown']:
                PRE_UP = None
            elif hl_item['name'] == 'drop':
                PRE_PICK = None
        except Exception as e:
            error_act_num += 1
            cprint(str(e), 'red')
            if 'Gripper is free' in str(e):
                PRE_PICK = None

            # cprint(str(e.__traceback__.tb_frame.f_globals['__file__']), 'red')  # File
            # cprint(str(e.__traceback__.tb_lineno), 'red')   # Line
            # if str(e).startswith('Error'):
            if str(e).startswith('Error') or ('Can not complete' in str(e)) or (
                    'list indices' in str(e)):
                error_act_num = 0
                # * Due to the actions_selected_num will be converted to the sample_weights,
                # * 0 means the sampling probability is 0.
                if data_idx is not None:
                    actions_selected_num[data_idx] = 0
            cprint('Reload previous state.'.format(e), 'green')
            if str(e).startswith('Logical Rule:'):
                # 2026-05-11 修改：逻辑前置条件错误通常还没改动物理状态，例如空夹爪 drop。
                # 这种错误不需要 restoreState，避免 Isaac/Replicator 状态下重复回滚导致卡住。
                env.resetDatapoint(config, previous_state_datapoint)
                continue
            # * Reload world state.
            try:
                _restore_with_timeout(config, state_id, previous_constraints, previous_state_datapoint)
            except Exception:
                break
            continue

        # * -----------------------------------------------------------
        # * Process 'climbUp', 'moveUp', 'pick'
        if (PRE_UP is None) and (hl_item['name'] in ['climbUp', 'moveUp']):
            PRE_UP = deepcopy(hl_item)
        elif (PRE_PICK is None) and (hl_item['name'] == 'pick'):
            PRE_PICK = deepcopy(hl_item)

        state_id, previous_constraints = env.saveState()  # * Save world state.
        _explore_debug('saveState after action: done')
        previous_constraints = deepcopy(previous_constraints)
        # * -----------------------------------------------------------
        # * Store data to graph.
        env_state_datapoint, rollback_datapoint = build_training_datapoint(
            config,
            state_id,
            previous_constraints,
            return_raw=True,
        )
        previous_state_datapoint = deepcopy(rollback_datapoint)
        assert len(env_state_datapoint.metrics) == len(env_state_datapoint.actions), \
            '[Error]: len(datapoint.metrics) != len(datapoint.actions)'
        graph.add_node(
            cursor_end,
            state=env_state_datapoint,
            id=cursor_end,
            parent_id=cursor_start,
            world_state=state_id,
            rollback_state=rollback_datapoint,
            pre_constraints=previous_constraints,
            child_actions=[],
        )  # * Add node.
        graph.add_weighted_edges_from([(cursor_start, cursor_end, 1.0), ])  # * Add edge.
        graph[cursor_start][cursor_end]['action'] = hl_actions  # * Set edge attribute.
        graph.nodes[cursor_start]['child_actions'].append(hl_item)
        _explore_debug(f'graph node {cursor_end} added')

        cursor_start = cursor_end
        cursor_end += 1
        step += 1
        # * -----------------------------------------------------------

    return graph, possible_hl_actions, actions_selected_num


def collect_data(config, first_explore_steps=10, random_select_node=99, explore_steps_per_node=10,
                 world_num=1, start_world_id=0):
    # * Load actions. #######参数设置########first_explore_steps=10, random_select_node=99, explore_steps_per_node=10,world_num=1, start_world_id=0
    possible_action_set, moveTo_set = load_Aall_and_split(config)
    actions_selected_num = torch.ones(len(possible_action_set))

    for i in range(start_world_id, world_num):
        # * -----------------------------------------------------------
        # * Create config.
        config.world = config.graph_world_name
        print('...' * 30)
        print('Creating new world in Isaac Lab (expff.usd) ...')
        print('...' * 30)
        time.sleep(3)
        env.start(config)

        # * -----------------------------------------------------------
        # * Create directed graph.
        DG = nx.DiGraph()
        node_id = 0  # * State (datapoint) in node 0 is empty.
        state_id, previous_constraints = env.saveState()
        env.initRootNode()  # * Add 'End' to the datapoint, use 'End' as node final state.
        # 2026-05-10 修改：根节点也写入 YOLO 观测状态；world_state 保留 Isaac 真值用于后续 restore。
        env_state_datapoint, rollback_datapoint = build_training_datapoint(
            config,
            state_id,
            previous_constraints,
            return_raw=True,
        )
        _explore_debug('root training datapoint ready')
        _explore_debug('root graph add_node: start')
        DG.add_node(
            node_id,
            state=env_state_datapoint,
            id=node_id,
            parent_id=None,
            world_state=state_id,
            rollback_state=rollback_datapoint,
            # * Pybullet state id. State id is used to restore the environment.
            pre_constraints=previous_constraints,
            child_actions=[]
        )  # * Add node.
        _explore_debug('root graph node added')

        # * -----------------------------------------------------------
        # * Collect data.
        # * 1. Explore from root node.
        _explore_debug('explore_from_node_i(root): start')
        graph, possible_hl_actions, actions_selected_num = explore_from_node_i(
            config, graph=DG, start_node_id=0, graph_last_node_id=0,
            explore_step=first_explore_steps,
            possible_hl_actions=(possible_action_set, moveTo_set),
            actions_selected_num=actions_selected_num
        )
        _explore_debug('explore_from_node_i(root): done')

        # * 2. Explore from random selected node.
        for _ in range(random_select_node):
            # * Must be here due to the node num is changed after exploration.
            nodes_list = list(DG.nodes)
            if len(nodes_list) <= 1:
                # 2026-05-11 修改：如果 root 探索阶段没有成功新增子节点，随机节点探索会反复选到 root。
                # root 没有 parent，旧逻辑会在 while True 中无限循环；这里直接跳过后续随机探索并保存已有 graph。
                _explore_debug('random node exploration skipped: graph has only root node')
                break

            # * Random choice node and reload world state.
            UP_FLAG = False
            selected_node_id = random.choice(nodes_list)
            select_attempts = 0
            while True:
                select_attempts += 1
                if select_attempts > len(nodes_list) * 2:
                    # 2026-05-11 修改：防止随机节点选择阶段因为没有可探索非 root 节点而卡死。
                    selected_node_id = None
                    break
                parent_node_id = graph.nodes[selected_node_id]['parent_id']
                if parent_node_id is not None:  # * parent_node_id is None in root node.
                    origin_hl_action = graph[parent_node_id][selected_node_id]['action']
                else:
                    selected_node_id = (selected_node_id + 1) % len(nodes_list)
                    continue
                origin_hl_item = origin_hl_action['actions'][0]
                if origin_hl_item['name'] in ['climbUp', 'moveUp']:
                    UP_FLAG = True
                    selected_node_id = (selected_node_id + 1) % len(nodes_list)
                elif UP_FLAG and (origin_hl_item['name'] in ['climbDown', 'moveDown']):
                    break
                else:
                    break
            if selected_node_id is None:
                _explore_debug('random node exploration skipped: no valid child node')
                break

            cprint('---' * 20, 'yellow')
            cprint('Explore from node: {}'.format(selected_node_id), 'yellow')
            graph, possible_hl_actions, actions_selected_num = explore_from_node_i(
                config,
                graph=DG,
                start_node_id=selected_node_id,
                graph_last_node_id=nodes_list[-1],
                explore_step=explore_steps_per_node,
                possible_hl_actions=possible_hl_actions,
                actions_selected_num=actions_selected_num
            )

        # * -----------------------------------------------------------
        # * Save graph.
        folder_name = os.path.join('data', config.domain, config.graph_world_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        data_num = len(os.listdir(folder_name))
        graph_save_path = folder_name + '/' + str(data_num) + '.graph'
        with open(graph_save_path, 'wb') as f:
            pickle.dump(DG, f)

        # * Destroy world.
        env.destroy()


if __name__ == '__main__':
    args = init_args()
    config = EnvironmentConfig(args)
    config.display = True

    collect_data(config)
    # load_Aall_and_split(config)
