import os
import torch
import pickle
import warnings
from tqdm import tqdm
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import get_model, load_model
from src.datasets.graph_dataset import GraphDataset_State

warnings.filterwarnings('ignore')


def _load_node_sequences(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _dataset_graph_refs(dataset_paths):
    refs = []
    for dataset_path in dataset_paths:
        for item in _load_node_sequences(dataset_path):
            graph_path = item.get('graph_path') if isinstance(item, dict) else None
            graph_file = item.get('graph_file') if isinstance(item, dict) else None
            world_name = item.get('world_name') if isinstance(item, dict) else None
            if graph_path is not None or graph_file is not None:
                refs.append(
                    {
                        'dataset_path': dataset_path,
                        'graph_path': os.path.abspath(graph_path) if graph_path is not None else None,
                        'graph_file': graph_file,
                        'world_name': world_name,
                    }
                )
    return refs


def _resolve_feature_graph_paths(config, dataset_paths):
    # 2026-05-12 修改：动作效果特征必须和 AFE 训练使用同一份 YOLO graph/split。
    # 默认读取 train_dataset.pkl 中记录的 graph_path；也可用 TAL_AFE_GRAPH_FILE/TAL_AFE_GRAPH_PATH 指定。
    refs = _dataset_graph_refs(dataset_paths)
    graph_path_env = os.environ.get('TAL_AFE_GRAPH_PATH') or os.environ.get('TAL_DATASET_GRAPH_PATH')
    graph_file_env = os.environ.get('TAL_AFE_GRAPH_FILE') or os.environ.get('TAL_DATASET_GRAPH_FILE')

    if graph_path_env:
        selected_paths = [
            os.path.abspath(item.strip())
            for item in graph_path_env.replace(',', ' ').split()
            if item.strip()
        ]
    elif graph_file_env:
        selected_paths = [
            os.path.abspath(os.path.join('./data', config.domain, config.graph_world_name, item.strip()))
            for item in graph_file_env.replace(',', ' ').split()
            if item.strip()
        ]
    else:
        selected_paths = sorted({ref['graph_path'] for ref in refs if ref['graph_path'] is not None})
        if not selected_paths:
            selected_paths = [
                os.path.abspath(os.path.join('./data', config.domain, config.graph_world_name, '11.graph'))
            ]

    missing_paths = [graph_path for graph_path in selected_paths if not os.path.exists(graph_path)]
    if missing_paths:
        raise FileNotFoundError(
            'Selected action-effect graph file(s) do not exist: {}'.format(', '.join(missing_paths))
        )

    selected_abs = set(selected_paths)
    selected_keys = {
        (os.path.basename(os.path.dirname(graph_path)), os.path.basename(graph_path))
        for graph_path in selected_paths
    }
    for ref in refs:
        ref_path = ref['graph_path']
        ref_key = (ref['world_name'], ref['graph_file'])
        if ref_path is not None and ref_path in selected_abs:
            continue
        if ref_key in selected_keys:
            continue
        raise RuntimeError(
            'Dataset {} references {}, but selected action-effect graph(s) are {}'.format(
                ref['dataset_path'], ref_path or ref_key, ', '.join(selected_paths)
            )
        )
    return selected_paths


def _prepare_selected_graphs_dir(selected_graph_paths):
    # 2026-05-12 修改：GraphDataset_State 会递归加载 graphs_dir 下所有 .graph。
    # 这里构造只包含选中 YOLO graph 的临时目录，避免生成动作效果特征时混入旧上帝视角 graph。
    selected_root = os.environ.get('TAL_AFE_SELECTED_GRAPHS_DIR', '/tmp/tal_afe_selected_graphs')
    os.makedirs(selected_root, exist_ok=True)
    for graph_path in selected_graph_paths:
        world_name = os.path.basename(os.path.dirname(graph_path))
        graph_file = os.path.basename(graph_path)
        dest_dir = os.path.join(selected_root, world_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, graph_file)
        if os.path.lexists(dest_path):
            if os.path.realpath(dest_path) == os.path.realpath(graph_path):
                continue
            if os.path.isdir(dest_path) and not os.path.islink(dest_path):
                raise RuntimeError('Cannot replace selected graph directory: {}'.format(dest_path))
            os.remove(dest_path)
        try:
            os.symlink(graph_path, dest_path)
        except OSError:
            import shutil
            shutil.copy2(graph_path, dest_path)
    return selected_root


def _ensure_graph_device(graph, device):
    graph = graph.to(device)
    if hasattr(graph, 'ndata'):
        for key, value in list(graph.ndata.items()):
            if torch.is_tensor(value) and value.device != device:
                graph.ndata[key] = value.to(device)
    return graph


def _move_sequence_to_device(config, graphSeq, goal2vec):
    if config.device is None:
        return graphSeq, goal2vec
    graphSeq = [_ensure_graph_device(graph, config.device) for graph in graphSeq]
    goal2vec = _ensure_graph_device(goal2vec, config.device)
    return graphSeq, goal2vec


def _log_devices_once(model, graph_a, graph_b):
    if getattr(_log_devices_once, "_done", False):
        return
    model_device = next(model.parameters()).device
    graph_a_device = graph_a.ndata['feat'].device
    graph_b_device = graph_b.ndata['feat'].device
    print(
        f"[AFE feature generation] model={model_device}, "
        f"state_a.feat={graph_a_device}, state_b.feat={graph_b_device}"
    )
    _log_devices_once._done = True


def generate_action_features(config, dataset, model, action_names, action_features):
    model.eval()

    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node) in tqdm(
            dataset, ncols=80):
        graphSeq, goal2vec = _move_sequence_to_device(config, graphSeq, goal2vec)
        graphSeq.append(goal2vec)
        with torch.no_grad():
            for i in range(len(graphSeq) - 1):
                tmp_idx = None
                if actionSeq[i] in action_names:
                    tmp_idx = action_names.index(actionSeq[i])
                else:
                    _log_devices_once(model, graphSeq[i], graphSeq[i + 1])
                    output, output_features = model(graphSeq[i], graphSeq[i + 1])
                    output_features = output_features.detach().cpu()
                    if tmp_idx is not None:
                        tmp_idx += 1
                        action_names.insert(tmp_idx, actionSeq[i])
                        action_features.insert(tmp_idx, output_features)
                    else:
                        action_names.append(actionSeq[i])
                        action_features.append(output_features)

    assert len(action_names) == len(action_features)
    print('Action num: {}'.format(len(action_names)))
    # return {'names': tuple(action_names), 'features': tuple(action_features)}
    return action_names, action_features


def generate_action_features_average(config, dataset, model, action_names, action_features):
    '''
    Averaging same actions' feature.
    action_names: ['Action_1', 'Action_2', ...]
    action_features: [Action_1_feature_list, Action_2_feature_list, ...]
    '''
    model.eval()

    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node) in tqdm(
            dataset, ncols=80):
        graphSeq, goal2vec = _move_sequence_to_device(config, graphSeq, goal2vec)
        graphSeq.append(goal2vec)

        for i in range(len(graphSeq) - 1):
            with torch.no_grad():
                _log_devices_once(model, graphSeq[i], graphSeq[i + 1])
                output, output_features = model(graphSeq[i], graphSeq[i + 1])
                output_features = output_features.detach().cpu()
            tmp_idx = None
            if actionSeq[i] in action_names:  # * Aciton already exists.
                tmp_idx = action_names.index(actionSeq[i])
                action_features[tmp_idx].append(output_features)
            else:
                if tmp_idx is not None:
                    tmp_idx += 1
                    action_names.insert(tmp_idx, actionSeq[i])
                    action_features.insert(tmp_idx, [output_features])  # * Feature list!!!
                else:
                    action_names.append(actionSeq[i])
                    action_features.append([output_features])

    assert len(action_names) == len(action_features)
    print('Action num: {}'.format(len(action_names)))
    # return {'names': tuple(action_names), 'features': tuple(action_features)}
    return action_names, action_features


if __name__ == '__main__':
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.model_name = ('AFE_MLP')
    args.model_name = ('AFE')
    config = EnvironmentConfig(args)

    # * ---------------------------------------------------------------
    # * Load data.
    train_data_path = './data/train_dataset.pkl'
    selected_graph_paths = _resolve_feature_graph_paths(config, [train_data_path])
    graphs_dir = _prepare_selected_graphs_dir(selected_graph_paths)
    print('Action-effect selected graph(s): {}'.format(', '.join(selected_graph_paths)))
    print('Action-effect graph loader dir: {}'.format(graphs_dir))
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    train_data_num = len(train_dataset)
    print('Train data num: {}'.format(train_data_num))

    # * ---------------------------------------------------------------
    # * Create model and load parameters.
    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model, optimizer, epoch, accuracy_list = load_model(
        config, seqTool + model.name + '_Trained', model
    )
    model = model.to(config.device)

    # * ---------------------------------------------------------------
    # * Generate action effect features.
    action_names = []
    action_features = []
    action_names, action_features = generate_action_features_average(config, train_dataset, model,
                                                                     action_names, action_features)
    # * Average action_features.
    avg_action_features = [sum(x) / len(x) for x in action_features]
    assert len(action_names) == len(action_features)
    print('Action num: {}'.format(len(action_names)))

    # * ---------------------------------------------------------------
    action_feature_dict = {'names': tuple(action_names), 'features': tuple(avg_action_features)}
    # * Save features.
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    features_save_path = './' + config.MODEL_SAVE_PATH + '/action_effect_features_avg.pkl'
    with open(features_save_path, 'wb') as f:
        pickle.dump(action_feature_dict, f)
