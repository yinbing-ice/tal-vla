"""
@File             : generate_and_split_dataset.py
@Project          : TAL
@Time             : 2021/11/22 9:00
@Author           : Xianqi ZHANG 
@Last Modify Time : 2024/02/29
@Desciption       : None
"""
import random
import pickle
import os
import colorama
import warnings
from tqdm import tqdm
from termcolor import cprint
from collections import deque
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.utils.graph import generate_node_list_from_graph, generate_node_list_from_graphs

colorama.init()
warnings.filterwarnings('ignore')


def calculate_actions(config, dataset, action_list):
    action_count = {}
    for item in tqdm(dataset, ncols=80):
        (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node) = item
        for action in actionSeq:
            if action not in action_list:
                action_list.append(action)
                if action['name'] not in action_count:
                    action_count[action['name']] = 1
                else:
                    action_count[action['name']] += 1
    # print(action_list[0])  # * {'name': 'pushTo', 'args': ['bottle_blue', 'mop']}
    cprint('--' * 20, 'green')
    cprint('Action num: {}'.format(len(action_list)), 'green')
    cprint(action_count, 'green')

    return action_list


def _resolve_selected_graph_paths(graphs_dir):
    # 2026-05-12 修改：默认只使用本次 YOLO exploration 生成的 world_expff/11.graph，
    # 避免旧的上帝视角 graph 混入 Generate dataset 阶段。
    # 如需更换 graph，可设置 TAL_DATASET_GRAPH_FILE=12.graph 或 TAL_DATASET_GRAPH_PATH=/abs/path/to/x.graph。
    graph_path_env = os.environ.get("TAL_DATASET_GRAPH_PATH", "").strip()
    graph_file_env = os.environ.get("TAL_DATASET_GRAPH_FILE", "11.graph").strip()

    selected_paths = []
    if graph_path_env:
        raw_paths = [item.strip() for item in graph_path_env.replace(",", " ").split() if item.strip()]
        selected_paths = [os.path.abspath(path) for path in raw_paths]
    elif graph_file_env:
        raw_files = [item.strip() for item in graph_file_env.replace(",", " ").split() if item.strip()]
        selected_paths = [
            os.path.abspath(os.path.join(graphs_dir, graph_file))
            for graph_file in raw_files
        ]

    missing_paths = [path for path in selected_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            "Selected graph file(s) do not exist: {}".format(", ".join(missing_paths))
        )
    return selected_paths


def _load_graphs(graphs_dir, selected_graph_paths=None):
    selected_set = None
    if selected_graph_paths is not None and len(selected_graph_paths) > 0:
        # 2026-05-12 修改：action 统计阶段也只加载选中的 graph，保持与数据集切分来源一致。
        selected_set = set(os.path.abspath(path) for path in selected_graph_paths)

    graphs_by_path = {}
    graphs_by_key = {}
    graphs_by_world = {}
    for world_name in os.listdir(graphs_dir):
        world_dir = os.path.join(graphs_dir, world_name)
        if not os.path.isdir(world_dir):
            continue
        graph_files = sorted(
            [file for file in os.listdir(world_dir) if file.endswith('.graph')]
        )
        if len(graph_files) == 0:
            continue
        for graph_file in graph_files:
            graph_path = os.path.abspath(os.path.join(world_dir, graph_file))
            if selected_set is not None and graph_path not in selected_set:
                continue
            with open(graph_path, 'rb') as f:
                dg = pickle.load(f)
            graphs_by_path[graph_path] = dg
            graphs_by_key[f"{world_name}/{graph_file}"] = dg
            if world_name not in graphs_by_world:
                graphs_by_world[world_name] = dg
    return graphs_by_path, graphs_by_key, graphs_by_world


def calculate_actions_from_node_sequences(graphs, node_sequences):
    graphs_by_path, graphs_by_key, graphs_by_world = graphs
    action_list = []
    action_count = {}
    for item in tqdm(node_sequences, ncols=80):
        world_name = item['world_name']
        node_seq = item['nodes']
        graph_path = item.get('graph_path')
        graph_file = item.get('graph_file')

        DG = None
        if graph_path is not None:
            DG = graphs_by_path.get(os.path.abspath(graph_path))
        if DG is None and graph_file is not None:
            DG = graphs_by_key.get(f"{world_name}/{graph_file}")
        if DG is None:
            DG = graphs_by_world[world_name]

        for idx in range(len(node_seq) - 1):
            action = DG[node_seq[idx]][node_seq[idx + 1]]['action']['actions'][0]
            if action not in action_list:
                action_list.append(action)
            action_count[action['name']] = action_count.get(action['name'], 0) + 1

    cprint('--' * 20, 'green')
    cprint('Action num: {}'.format(len(action_list)), 'green')
    cprint(action_count, 'green')
    return action_list


def split_sequences_adaptive(actions_num, training_set, test_set, training_sequence_num,
                             val_sequence_num, test_sequence_num):
    train_data, val_data, test_data = [], [], []

    for i in range(0, len(actions_num)):
        data_d = deque(actions_num[i])
        data_num = len(data_d)
        if data_num == 0:
            continue
        for _ in range(5):
            random.shuffle(data_d)
            rotate_num = random.randint(max(1, int(max(data_num, 1) / 5)), data_num)
            data_d.rotate(rotate_num)
            actions_num[i] = list(data_d)

        if i in training_set:
            take_num = min(training_sequence_num, len(actions_num[i]))
            train_data.extend(actions_num[i][:take_num])
        elif i in test_set:
            available = len(actions_num[i])
            val_take = min(val_sequence_num, max(available // 3, 1))
            test_take = min(test_sequence_num, max(available - val_take, 0))
            val_data.extend(actions_num[i][:val_take])
            test_data.extend(actions_num[i][val_take: val_take + test_take])

    return train_data, val_data, test_data


if __name__ == '__main__':
    args = init_args()
    # args.world = 'src/envs/jsons/home_worlds/world_home1.json'
    config = EnvironmentConfig(args)

    # * Generate data from selected graph(s).
    graphs_dir = 'data/{}/{}/'.format(config.domain, config.graph_world_name)
    selected_graph_paths = _resolve_selected_graph_paths(graphs_dir)
    min_sequence_length = 2  # * Three node (2 actions). Remove only 1 step.
    max_sequence_length = 11
    if selected_graph_paths:
        # 2026-05-12 修改：针对新生成的 YOLO graph 逐个生成 node sequence；
        # item 中会保留 graph_path/graph_file，后续训练数据读取时能精确定位来源 graph。
        node_sequences = []
        for graph_path in selected_graph_paths:
            cprint('Using selected graph: {}'.format(graph_path), 'green')
            node_sequences.extend(
                generate_node_list_from_graph(
                    config,
                    graph_path,
                    max_sequence_length=max_sequence_length,
                    start_index=0,
                    check_json=True,
                    STATE_FORMAT_GOAL=True
                )
            )
    else:
        node_sequences = generate_node_list_from_graphs(
            config,
            graphs_dir,
            max_sequence_length=max_sequence_length,
            start_index=0,
            check_json=True,
            STATE_FORMAT_GOAL=True
        )
    node_sequences = [
        item for item in node_sequences
        if len(item['nodes']) >= min_sequence_length and len(item['nodes']) <= max_sequence_length
    ]
    data_len = len(node_sequences)
    print('Total data num: {}'.format(data_len))
    if data_len == 0:
        raise RuntimeError(
            'No node sequences were generated. Please check the exploration graph or '
            'lower the min/max sequence length constraints.'
        )

    # * ----------------------------------------------------------------
    # # * 1. Random split.
    # # * Data split: 8: 1: 1
    # random.shuffle(node_sequences)
    # td_len = len(node_sequences)
    # point_1 = int(td_len * 0.6)
    # point_2 = int(td_len * 0.8)
    # train_data = node_sequences[0: point_1]
    # val_data = node_sequences[point_1: point_2]
    # test_data = node_sequences[point_2:]
    # * ----------------------------------------------------------------
    # # * 2. Stratified split.
    # * Data split: 8: 1: 1
    # * Calculate data length.
    # * The data which has same actions num is saved in the same list.
    actions_num = [list() for _ in range(max_sequence_length)]
    for data in node_sequences:
        act_num = len(data['nodes']) - 1
        actions_num[act_num].append(data)

    print('--' * 10)
    for i in range(len(actions_num)):
        print('{} : {}'.format(i, len(actions_num[i])))
    print('--' * 10)

    # * ----------------------------------------------------------------
    # * Generate data.
    # * Shuffle.
    training_sequence_num = 800  # * Num for one type length sequence.
    val_sequence_num = 500
    test_sequence_num = 500

    training_set = [1, 2, 3]
    test_set = [4, 5, 6, 7, 8, 9, 10]

    train_data, val_data, test_data = split_sequences_adaptive(
        actions_num,
        training_set,
        test_set,
        training_sequence_num,
        val_sequence_num,
        test_sequence_num,
    )

    print('Train num: {}'.format(len(train_data)))
    print('Val num: {}'.format(len(val_data)))
    print('Test num: {}'.format(len(test_data)))

    # * ----------------------------------------------------------------
    # * Save data.
    dataset_save_path = './data/graph_dataset.pkl'
    train_dataset_save_path = './data/train_dataset.pkl'
    val_dataset_save_path = './data/val_dataset.pkl'
    test_dataset_save_path = './data/test_dataset.pkl'

    with open(dataset_save_path, 'wb') as f:
        pickle.dump(node_sequences, f)
    with open(train_dataset_save_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_dataset_save_path, 'wb') as f:
        pickle.dump(val_data, f)
    with open(test_dataset_save_path, 'wb') as f:
        pickle.dump(test_data, f)

    # * ----------------------------------------------------------------
    # * Check actions in data with a lightweight graph reader.
    graphs = _load_graphs('./data/{}/'.format(config.domain), selected_graph_paths)

    print('Training data:')
    action_list_train = calculate_actions_from_node_sequences(graphs, train_data)
    print('Validation data:')
    action_list_val = calculate_actions_from_node_sequences(graphs, val_data)
    print('Test data:')
    action_list_test = calculate_actions_from_node_sequences(graphs, test_data)

    # * Save features.
    action_list_train_save_path = './' + config.MODEL_SAVE_PATH + 'action_list_train_dataset.pkl'
    os.makedirs(os.path.dirname(action_list_train_save_path), exist_ok=True)
    with open(action_list_train_save_path, 'wb') as f:
        pickle.dump(action_list_train, f)

    # action_list_val_save_path = './' + config.MODEL_SAVE_PATH + 'action_list_val_dataset.pkl'
    # with open(action_list_val_save_path, 'wb') as f:
    #     pickle.dump(action_list_val, f)
    #
    # action_list_test_save_path = './' + config.MODEL_SAVE_PATH + 'action_list_test_dataset.pkl'
    # with open(action_list_test_save_path, 'wb') as f:
    #     pickle.dump(action_list_test, f)
