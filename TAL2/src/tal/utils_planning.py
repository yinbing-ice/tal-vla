import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import islice
import traceback

from src.envs import approx
from src.tal.action_proposal_network import vec2action_grammatical
from src.tal.scene_graph_translator import get_current_scene_graph_json, \
    translate_instruction_to_goal_state_graph


def _safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


def get_action_pred_with_model_actions(
        config,
        model_action,
        action_effect_features,
        graph,
        goal2vec
):
    model_action.eval()
    with torch.no_grad():
        action_pred = model_action(graph, goal2vec)
        action_array = list(action_pred[:len(config.possibleActions)])
        curr_1 = len(config.possibleActions)
        curr_2 = len(config.possibleActions) + config.num_objects
        curr_3 = len(config.possibleActions) + config.num_objects + config.num_objects
        object1_array = list(action_pred[curr_1:curr_2])
        object2_array = list(action_pred[curr_2:curr_3])
        state_array = list(action_pred[curr_3:])
    # * ---------------------------------------------------------------
    action_names = action_effect_features['names']
    output = torch.zeros((1, len(action_names)), device=config.device)
    for i, act_name in enumerate(
            action_names):  # * {'name': 'pushTo', 'args': ['vacuum', 'sponge']}
        action_name = act_name['name']
        action_args = act_name['args']
        # * Sum all probs.
        action_name_prob = action_array[config.possibleActions.index(action_name)]
        output[0][i] += action_name_prob
        if len(action_args) == 1:
            action_obj_1 = object1_array[config.object2idx[action_args[0]]]
            output[0][i] += action_obj_1
        elif len(action_args) == 2:
            action_args_1 = object1_array[config.object2idx[action_args[0]]]
            output[0][i] += action_args_1
            action_args_2 = object2_array[config.object2idx[action_args[1]]]
            output[0][i] += action_args_2

    return output


def get_action_pred_with_model_extract_feature(
        config,
        model_extract_feature,
        action_effect_features,
        graph,
        goal2vec
):
    model_extract_feature.eval()
    action_features = action_effect_features['features']
    # * Convert to a tensor: each row is a [1, 128 * 32] action feature.
    # * squeeze(): [828, 1, 128 * 32] --> [828, 128 * 32], [action_num, feature_dim]
    action_features_tensor = torch.stack(action_features).squeeze(1)
    act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)

    # * Extract current task features.
    with torch.no_grad():
        _, current_task_feature = model_extract_feature(graph, goal2vec)  # * [1, 128 * 32]

    if config.device is not None:
        act_generalized_inverse_mat = torch.tensor(act_generalized_inverse_mat,
                                                   device=config.device)
    output = torch.matmul(current_task_feature, act_generalized_inverse_mat)  # * !!!
    output = F.softmax(output)
    return output


def check_policy_with_model_actions(
        config,
        model_action,
        data_item,
        world_num,
        INIT_DATAPOINT,
        STATE_FORMAT_GOAL
):
    model_action.eval()
    # * ---------------------------------------------------------------
    # * Get data.
    if STATE_FORMAT_GOAL:  # * GraphDataset_State
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
    else:  # * GraphDataset
        graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

    predActionSeq = []  # * Initialize
    graphSeq_t = []
    # * Initialize environment.
    if INIT_DATAPOINT:
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=start_node)
        graphSeq_t.append(graphSeq[0])
    else:
        # * Use wold_home1 to test.
        # world_num = 1
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
        init_g = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            init_g = init_g.to(config.device)
        graphSeq_t.append(init_g)

    # * Test model.
    y_pred_list = []
    while True:
        with torch.no_grad():
            output = model_action(graphSeq_t[-1], goal2vec)
            y_pred_list.append(output)
        y_pred = y_pred_list[-1]
        action_pred = vec2action_grammatical(
            config, y_pred, config.num_objects, len(config.possibleStates), config.idx2object
        )
        # * !!!
        res, g, err = approx.execAction(config, action_pred, config.embeddings)
        predActionSeq.append(action_pred)
        if g is not None and config.device is not None:
            g = g.to(config.device)
        graphSeq_t.append(g)

        # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
        if res:
            return 'Correct', len(actionSeq), len(predActionSeq)
        elif err == '' and len(predActionSeq) > 60:
            return 'Incorrect', None, None
        elif err != '':
            return 'Error', None, None


def process_feature_with_pca(A, principal_directions=None, threshold=0.05, q_value=None):
    if principal_directions is not None:
        if torch.is_tensor(principal_directions) and principal_directions.device != A.device:
            principal_directions = principal_directions.to(A.device)
        principal_directions = principal_directions.to(device=A.device, dtype=A.dtype)
        ret_tensor = torch.matmul(A, principal_directions)
        ret_v = principal_directions
    else:
        max_rank = min(A.shape[0], A.shape[1])
        if max_rank <= 0:
            raise ValueError('PCA input must have non-zero shape, got {}.'.format(tuple(A.shape)))
        if q_value is None:
            q_value = max_rank
        else:
            q_value = max(1, min(int(q_value), max_rank))
        U, S, V = torch.pca_lowrank(A, q=q_value)
        if V.device != A.device:
            V = V.to(A.device)
        ret_tensor = torch.matmul(A, V)
        ret_v = V
    # ret_v[ret_v < threshold] = 0
    return ret_tensor, ret_v


def check_policy_with_feature_pca_order(
        config,
        model_action,
        model_extract_feature,
        data_item,
        world_num,
        action_effect_features,
        INIT_DATAPOINT,
        STATE_FORMAT_GOAL,
        candidate_action_num=20,
        select_from_candidate=10,
        trajectory_length=60
):
    model_action.eval()
    model_extract_feature.eval()
    # * Get data.
    if STATE_FORMAT_GOAL:  # * GraphDataset_State
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
    else:  # * GraphDataset
        graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    # * Convert to a tensor: each row is a [1, 512] action feature.
    # * squeeze(): [828, 1, 512] --> [828, 512], [action_num, feature_dim]
    action_features_tensor = torch.stack(action_features).squeeze(1)
    if config.device is not None and action_features_tensor.device != config.device:
        action_features_tensor = action_features_tensor.to(config.device)
    action_features_tensor, principal_directions = process_feature_with_pca(action_features_tensor,
                                                                            q_value=500)

    predActionSeq = []  # * Initialize
    graphSeq_t = []
    # * Initialize environment.
    if INIT_DATAPOINT:
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=start_node)
        graphSeq_t.append(graphSeq[0])
    else:
        # * Use wold_home1 to test.
        # world_num = 1
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
        init_g = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            init_g = init_g.to(config.device)
        graphSeq_t.append(init_g)

    # * Generate actions and execute.
    selected_actions = []
    while True:
        # * Extract current task features.
        # if len(selected_actions) == 0:
        if len(selected_actions) <= (candidate_action_num - select_from_candidate):
            selected_actions.clear()
            with torch.no_grad():
                _, current_task_feature = model_extract_feature(graphSeq_t[-1],
                                                                goal2vec)  # * [1, 512]
                current_task_feature, _ = process_feature_with_pca(current_task_feature,
                                                                   principal_directions)

            # * 01. Generation: Direct.
            # * Select actions.
            act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
            if act_generalized_inverse_mat.device != current_task_feature.device:
                act_generalized_inverse_mat = act_generalized_inverse_mat.to(current_task_feature.device)
            output = torch.matmul(current_task_feature, act_generalized_inverse_mat)  # * !!!

            output = output.squeeze(0)
            tpk = torch.topk(output, candidate_action_num)
            action_idx = tpk.indices
            action_idx = torch.sort(action_idx).values
            for idx in action_idx:
                selected_actions.append(action_names[idx])

        actions_prob = get_action_pred_with_model_actions(
            config, model_action, action_effect_features, graphSeq_t[-1], goal2vec
        )

        selected_actions_prob = []
        for act in selected_actions:
            idx = action_names.index(act)
            prob = actions_prob[0][idx]
            selected_actions_prob.append(prob)
        max_prob = max(selected_actions_prob)
        action_current_select = selected_actions[selected_actions_prob.index(max_prob)]
        selected_actions.remove(action_current_select)  # * Remove selected actions.

        # * !!!
        res, g, err = approx.execAction(config, action_current_select, config.embeddings)
        predActionSeq.append(action_current_select)
        if g is not None and config.device is not None:
            g = g.to(config.device)
        graphSeq_t.append(g)

        # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
        if res:
            return 'Correct', len(actionSeq), len(predActionSeq)
        elif err == '' and len(predActionSeq) > trajectory_length:
            return 'Incorrect', None, None
        elif err != '':
            return 'Error', None, None


def check_policy_with_feature(
        config,
        model_action,
        model_extract_feature,
        data_item,
        world_num,
        action_effect_features,
        INIT_DATAPOINT,
        STATE_FORMAT_GOAL,
        candidate_action_num=20,
        select_from_candidate=10,
        trajectory_length=60
):
    """
    Original feature dimension, without PCA.
    """
    model_action.eval()
    model_extract_feature.eval()
    # * Get data.
    if STATE_FORMAT_GOAL:  # * GraphDataset_State
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
    else:  # * GraphDataset
        graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    # * Convert to a tensor: each row is a [1, 512] action feature.
    # * squeeze(): [828, 1, 512] --> [828, 512], [action_num, feature_dim]
    action_features_tensor = torch.stack(action_features).squeeze(1)

    predActionSeq = []  # * Initialize
    graphSeq_t = []
    # * Initialize environment.
    if INIT_DATAPOINT:
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=start_node)
        graphSeq_t.append(graphSeq[0])
    else:
        # * Use wold_home1 to test.
        # world_num = 1
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
        init_g = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            init_g = init_g.to(config.device)
        graphSeq_t.append(init_g)

    # * Generate actions and execute.
    selected_actions = []
    while True:
        # * Extract current task features.
        # if len(selected_actions) == 0:
        if len(selected_actions) <= (candidate_action_num - select_from_candidate):
            selected_actions.clear()
            with torch.no_grad():
                _, current_task_feature = model_extract_feature(graphSeq_t[-1],
                                                                goal2vec)  # * [1, 512]

            # * 01. Generation: Direct.
            # * Select actions.
            act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
            output = torch.matmul(current_task_feature, act_generalized_inverse_mat)  # * !!!
            output = output.squeeze(0)
            tpk = torch.topk(output, candidate_action_num)
            action_idx = tpk.indices
            action_idx = torch.sort(action_idx).values
            for idx in action_idx:
                selected_actions.append(action_names[idx])

        actions_prob = get_action_pred_with_model_actions(
            config, model_action, action_effect_features, graphSeq_t[-1], goal2vec
        )

        selected_actions_prob = []
        for act in selected_actions:
            idx = action_names.index(act)
            prob = actions_prob[0][idx]
            selected_actions_prob.append(prob)
        max_prob = max(selected_actions_prob)
        action_current_select = selected_actions[selected_actions_prob.index(max_prob)]
        selected_actions.remove(action_current_select)  # * Remove selected actions.

        # * !!!
        res, g, err = approx.execAction(config, action_current_select, config.embeddings)
        predActionSeq.append(action_current_select)
        if g is not None and config.device is not None:
            g = g.to(config.device)
        graphSeq_t.append(g)

        # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
        if res:
            return 'Correct', len(actionSeq), len(predActionSeq)
        elif err == '' and len(predActionSeq) > trajectory_length:
            return 'Incorrect', None, None
        elif err != '':
            return 'Error', None, None


def test_policy_with_action_and_memory_model(
        config,
        dataset,
        model_action,
        model_extract_feature,
        action_effect_features,
        TQDM=True,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True
):
    """
    model_action: predict action
    model_extract_feature:
    output_action_prob + lambda * output_memory_prob
    """
    model_action.eval()
    model_extract_feature.eval()
    action_names = action_effect_features['names']
    correct, incorrect, error = 0, 0, 0
    lenHuman, lenModel = [], []
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])
    data_num = {}
    data_correct_num = {}
    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        # * Get data.
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        # * Store action length to dict.
        if str(len(actionSeq)) in data_num:
            data_num[str(len(actionSeq))] += 1
        else:
            data_num[str(len(actionSeq))] = 1
        if str(len(actionSeq)) not in data_correct_num:
            data_correct_num[str(len(actionSeq))] = 0

        world_num = _safe_numeric_suffix(world_name, default=0)
        plan_len = len(actionSeq)
        lenHuman.append(plan_len)
        predActionSeq = []  # * Initialize
        graphSeq_t = []
        # * Initialize environment.
        if INIT_DATAPOINT:
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                              INPUT_DATAPOINT=start_node)
            graphSeq_t.append(graphSeq[0])
        else:
            # * Use wold_home1 to test.
            # world_num = 1
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
            init_g = approx.getInitializeDGLGraph(config)
            if config.device is not None:
                init_g = init_g.to(config.device)
            graphSeq_t.append(init_g)

        # * Test model.
        y_pred_list = []
        while True:
            output_action = get_action_pred_with_model_actions(
                config, model_action, action_effect_features, graphSeq_t[-1], goal2vec
            )
            output_memory = get_action_pred_with_model_extract_feature(
                config, model_extract_feature, action_effect_features, graphSeq_t[-1], goal2vec
            )
            output_memory -= output_memory.min()
            output_memory = output_memory / output_memory.max() * output_action.max()
            gamma = 1.0
            y_pred = output_action + gamma * output_memory
            # y_pred = output_action
            # y_pred = output_memory
            action_pred = action_names[y_pred.argmax()]
            y_pred_list.append(action_pred)

            # * !!!
            res, g, err = approx.execAction(config, action_pred, config.embeddings)
            predActionSeq.append(action_pred)
            if g is not None and config.device is not None:
                g = g.to(config.device)
            graphSeq_t.append(g)

            # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
            if res:
                data_correct_num[str(len(actionSeq))] += 1
                correct += 1
                lenModel.append(len(predActionSeq))
                break
            elif err == '' and len(predActionSeq) > 60:
                incorrect += 1
                break
            elif err != '':
                error += 1
                break

    # * Print results.
    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def check_policy_with_model_extract_feature_pca_wo_order(
        config,
        model_extract_feature,
        data_item,
        world_num,
        action_effect_features,
        INIT_DATAPOINT,
        STATE_FORMAT_GOAL,
        candidate_action_num=20,
        select_from_candidate=10,
        trajectory_length=60
):
    model_extract_feature.eval()
    # * Get data.
    if STATE_FORMAT_GOAL:  # * GraphDataset_State
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
    else:  # * GraphDataset
        graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    # * Convert to a tensor: each row is a [1, 4096] action feature.
    # * squeeze(): [828, 1, 4096] --> [828, 4096], [action_num, feature_dim]
    action_features_tensor = torch.stack(action_features).squeeze(1)
    action_features_tensor, principal_directions = process_feature_with_pca(action_features_tensor,
                                                                            q_value=500)
    # cprint('Feature shape after PCA: {}'.format(action_features_tensor.shape), 'green')
    # act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)

    predActionSeq = []  # * Initialize
    graphSeq_t = []
    # * Initialize environment.
    if INIT_DATAPOINT:
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=start_node)
        graphSeq_t.append(graphSeq[0])
    else:
        # * Use wold_home1 to test.
        # world_num = 1
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
        init_g = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            init_g = init_g.to(config.device)
        graphSeq_t.append(init_g)

    # * Generate actions and execute.
    while True:
        # * Extract current task features.
        with torch.no_grad():
            _, current_task_feature = model_extract_feature(graphSeq_t[-1], goal2vec)  # * [1, 512]
            current_task_feature, _ = process_feature_with_pca(current_task_feature,
                                                               principal_directions)

        # * 01. Generation: Direct.
        selected_actions = []
        # * Select actions.
        act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
        output = torch.matmul(current_task_feature, act_generalized_inverse_mat)  # * !!!

        output = output.squeeze(0)
        tpk = torch.topk(output, 15)
        action_idx = tpk.indices
        action_idx = torch.sort(action_idx).values  # * Ordered by indices.
        for idx in action_idx:
            selected_actions.append(action_names[idx])

        # * !!!
        for action_pred in selected_actions:
            res, g, err = approx.execAction(config, action_pred, config.embeddings)
            predActionSeq.append(action_pred)
            if g is not None and config.device is not None:
                g = g.to(config.device)
            graphSeq_t.append(g)

            # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
            if res:
                return 'Correct', len(actionSeq), len(predActionSeq)
            elif err == '' and len(predActionSeq) > 60:
                return 'Incorrect', None, None
            elif err != '':
                return 'Error', None, None


def test_policy_with_action_effect_features(
        config,
        dataset,
        model_action,
        model_extract_feature,
        action_effect_features,
        multiscale_pool=None,
        TQDM=True,
        ONLY_ACTION_MODEL=False,
        ONLY_FEATURE_MODEL=False,
        WITH_PCA=True,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True,
        IGNORE_ACTION_MODEL=False,
        MAX_SAMPLES=-1,
        PRINT_SAMPLE_INFO=False,
        PROGRESS_DESC='Policy Testing'
):
    """
    model_action: predict action
    model_extract_feature:
    """
    model_action.eval()
    if not ONLY_ACTION_MODEL:
        model_extract_feature.eval()

    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])
    data_num = {}
    data_correct_num = {}
    dataset_iter = iter(dataset)
    if MAX_SAMPLES is not None and MAX_SAMPLES >= 0:
        dataset_iter = islice(dataset_iter, MAX_SAMPLES)
    if TQDM:
        total = None
        if hasattr(dataset, '__len__'):
            total = len(dataset) if MAX_SAMPLES is None or MAX_SAMPLES < 0 else min(len(dataset), MAX_SAMPLES)
        data_container = tqdm(dataset_iter, desc=PROGRESS_DESC, ncols=100, total=total)
    else:
        data_container = dataset_iter
    for sample_idx, data_item in enumerate(data_container):
        try:
            # * Get data.
            if STATE_FORMAT_GOAL:  # * GraphDataset_State
                graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
            else:  # * GraphDataset
                graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

            if PRINT_SAMPLE_INFO:
                print(
                    '[Sample {:>4}] world={} | plan_len={}'.format(
                        sample_idx, world_name, len(actionSeq)
                    )
                )

            # * Store action length to dict.
            if str(len(actionSeq)) in data_num:
                data_num[str(len(actionSeq))] += 1
            else:
                data_num[str(len(actionSeq))] = 1
            if str(len(actionSeq)) not in data_correct_num:
                data_correct_num[str(len(actionSeq))] = 0

            world_num = _safe_numeric_suffix(world_name, default=0)
            plan_len = len(actionSeq)
            lenHuman.append(plan_len)
            if ONLY_ACTION_MODEL:
                res, len_acts, len_pred_acts = check_policy_with_model_actions(
                    config, model_action, data_item, world_num, INIT_DATAPOINT, STATE_FORMAT_GOAL
                )
                if res == 'Correct':
                    data_correct_num[str(len_acts)] += 1
                    correct += 1
                    lenModel.append(len_pred_acts)
                elif res == 'Incorrect':
                    incorrect += 1
                elif res == 'Error':
                    error += 1

            elif ONLY_FEATURE_MODEL:
                res, len_acts, len_pred_acts = check_policy_with_model_extract_feature_pca_wo_order(
                    config,
                    model_extract_feature,
                    data_item,
                    world_num,
                    action_effect_features,
                    INIT_DATAPOINT,
                    STATE_FORMAT_GOAL
                )
                if res == 'Correct':
                    data_correct_num[str(len_acts)] += 1
                    correct += 1
                    lenModel.append(len_pred_acts)
                elif res == 'Incorrect':
                    incorrect += 1
                elif res == 'Error':
                    error += 1

            else:

                if not IGNORE_ACTION_MODEL:
                    res, len_acts, len_pred_acts = check_policy_with_model_actions(
                        config, model_action, data_item, world_num, INIT_DATAPOINT, STATE_FORMAT_GOAL
                    )
                else:
                    res = 'False'

                if res == 'Correct':
                    data_correct_num[str(len_acts)] += 1
                    correct += 1
                    lenModel.append(len_pred_acts)
                else:
                    tmp_res = None
                    if multiscale_pool is None:
                        multiscale_pool = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5),
                                           (30, 10)]
                    for (tmp_cdd_act_num, tmp_slt_act_num) in multiscale_pool:
                        if WITH_PCA:
                            res, len_acts, len_pred_acts = check_policy_with_feature_pca_order(
                                config,
                                model_action,
                                model_extract_feature,
                                data_item,
                                world_num,
                                action_effect_features,
                                INIT_DATAPOINT,
                                STATE_FORMAT_GOAL,
                                candidate_action_num=tmp_cdd_act_num,
                                select_from_candidate=tmp_slt_act_num
                            )
                        else:
                            res, len_acts, len_pred_acts = check_policy_with_feature(
                                config,
                                model_action,
                                model_extract_feature,
                                data_item,
                                world_num,
                                action_effect_features,
                                INIT_DATAPOINT,
                                STATE_FORMAT_GOAL,
                                candidate_action_num=tmp_cdd_act_num,
                                select_from_candidate=tmp_slt_act_num
                            )

                        tmp_res = res
                        if res == 'Correct':
                            data_correct_num[str(len_acts)] += 1
                            correct += 1
                            lenModel.append(len_pred_acts)
                            break
                    if tmp_res == 'Incorrect':
                        incorrect += 1
                    elif tmp_res == 'Error':
                        error += 1
        except Exception:
            print(
                '[test_policy_with_action_effect_features] sample={} world={} plan_len={}'.format(
                    sample_idx,
                    world_name if 'world_name' in locals() else 'unknown',
                    len(actionSeq) if 'actionSeq' in locals() else 'unknown'
                )
            )
            traceback.print_exc()
            raise

    # * Print results.
    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


# * Generalization data test.
def test_policy_with_action_effect_features_generalization(
        config,
        dataset,
        model_action,
        model_extract_feature,
        action_effect_features,
        TQDM=True,
        ONLY_ACTION_MODEL=False,
        ONLY_FEATURE_MODEL=False,
        WITH_PCA=True,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True
):
    """
    model_action: predict action
    model_extract_feature:
    """
    model_action.eval()
    model_extract_feature.eval()
    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])

    data_num = {}
    data_correct_num = {}
    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        # * Get data.
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        # * Store action length to dict.
        if goal_json['goal-name'] in data_num:
            data_num[goal_json['goal-name']] += 1
        else:
            data_num[goal_json['goal-name']] = 1
        if goal_json['goal-name'] not in data_correct_num:
            data_correct_num[goal_json['goal-name']] = 0

        world_num = _safe_numeric_suffix(world_name, default=0)
        plan_len = len(actionSeq)
        lenHuman.append(plan_len)
        if ONLY_ACTION_MODEL:
            res, len_acts, len_pred_acts = check_policy_with_model_actions(
                config, model_action, data_item, world_num, INIT_DATAPOINT, STATE_FORMAT_GOAL
            )
            if res == 'Correct':
                data_correct_num[goal_json['goal-name']] += 1
                correct += 1
                lenModel.append(len_pred_acts)
            elif res == 'Incorrect':
                incorrect += 1
            elif res == 'Error':
                error += 1

        elif ONLY_FEATURE_MODEL:
            res, len_acts, len_pred_acts = check_policy_with_model_extract_feature_pca_wo_order(
                config,
                model_extract_feature,
                data_item,
                world_num,
                action_effect_features,
                INIT_DATAPOINT,
                STATE_FORMAT_GOAL
            )
            if res == 'Correct':
                data_correct_num[goal_json['goal-name']] += 1
                correct += 1
                lenModel.append(len_pred_acts)
            elif res == 'Incorrect':
                incorrect += 1
            elif res == 'Error':
                error += 1

        else:
            res, len_acts, len_pred_acts = check_policy_with_model_actions(
                config, model_action, data_item, world_num, INIT_DATAPOINT, STATE_FORMAT_GOAL
            )
            if res == 'Correct':
                data_correct_num[goal_json['goal-name']] += 1
                correct += 1
                lenModel.append(len_pred_acts)
            else:
                tmp_res = None
                # multiscale_pool = [(5, 3), (10, 3), (10, 5), (20, 5), (20, 10), (20, 15)]
                multiscale_pool = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
                for (tmp_cdd_act_num, tmp_slt_act_num) in multiscale_pool:
                    if WITH_PCA:
                        res, len_acts, len_pred_acts = check_policy_with_feature_pca_order(
                            config,
                            model_action,
                            model_extract_feature,
                            data_item,
                            world_num,
                            action_effect_features,
                            INIT_DATAPOINT,
                            STATE_FORMAT_GOAL,
                            candidate_action_num=tmp_cdd_act_num,
                            select_from_candidate=tmp_slt_act_num
                        )
                    else:
                        res, len_acts, len_pred_acts = check_policy_with_feature(
                            config,
                            model_action,
                            model_extract_feature,
                            data_item,
                            world_num,
                            action_effect_features,
                            INIT_DATAPOINT,
                            STATE_FORMAT_GOAL,
                            candidate_action_num=tmp_cdd_act_num,
                            select_from_candidate=tmp_slt_act_num
                        )

                    tmp_res = res
                    if res == 'Correct':
                        data_correct_num[goal_json['goal-name']] += 1
                        correct += 1
                        lenModel.append(len_pred_acts)
                        break
                if tmp_res == 'Incorrect':
                    incorrect += 1
                elif tmp_res == 'Error':
                    error += 1

    # * Print results.
    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    # key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def test_policy_with_only_pool(
        config,
        dataset,
        model_action,
        model_extract_feature,
        action_effect_features,
        TQDM=True,
        WITH_PCA=True,
        POOL_SET=None,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True
):
    """
    model_action: predict action
    model_extract_feature:
    """
    model_action.eval()
    model_extract_feature.eval()

    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])
    data_num = {}
    data_correct_num = {}
    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        # * Get data.
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        # * Store action length to dict.
        if str(len(actionSeq)) in data_num:
            data_num[str(len(actionSeq))] += 1
        else:
            data_num[str(len(actionSeq))] = 1
        if str(len(actionSeq)) not in data_correct_num:
            data_correct_num[str(len(actionSeq))] = 0

        world_num = _safe_numeric_suffix(world_name, default=0)
        plan_len = len(actionSeq)
        lenHuman.append(plan_len)
        tmp_res = None
        if POOL_SET is None:
            multiscale_pool = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
        else:
            multiscale_pool = POOL_SET
        for (tmp_cdd_act_num, tmp_slt_act_num) in multiscale_pool:
            if WITH_PCA:
                res, len_acts, len_pred_acts = check_policy_with_feature_pca_order(
                    config,
                    model_action,
                    model_extract_feature,
                    data_item,
                    world_num,
                    action_effect_features,
                    INIT_DATAPOINT,
                    STATE_FORMAT_GOAL,
                    candidate_action_num=tmp_cdd_act_num,
                    select_from_candidate=tmp_slt_act_num
                )
            else:
                res, len_acts, len_pred_acts = check_policy_with_feature(
                    config, model_action,
                    model_extract_feature,
                    data_item,
                    world_num,
                    action_effect_features,
                    INIT_DATAPOINT,
                    STATE_FORMAT_GOAL,
                    candidate_action_num=tmp_cdd_act_num,
                    select_from_candidate=tmp_slt_act_num
                )

            tmp_res = res
            if res == 'Correct':
                data_correct_num[str(len_acts)] += 1
                correct += 1
                lenModel.append(len_pred_acts)
                break
            if tmp_res == 'Incorrect':
                incorrect += 1
            elif tmp_res == 'Error':
                error += 1

    # * Print results.
    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def qualitative_policy_with_feature_pca_order(
        config,
        model_action,
        model_extract_feature,
        data_item,
        world_num,
        action_effect_features,
        INIT_DATAPOINT,
        STATE_FORMAT_GOAL,
        candidate_action_num=20,
        select_from_candidate=10,
        trajectory_length=60
):
    model_action.eval()
    model_extract_feature.eval()
    # * Get data.
    if STATE_FORMAT_GOAL:  # * GraphDataset_State
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
    else:  # * GraphDataset
        graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    # * Convert to a tensor: each row is a [1, 512] action feature.
    # * squeeze(): [828, 1, 512] --> [828, 512], [action_num, feature_dim]
    action_features_tensor = torch.stack(action_features).squeeze(1)
    if config.device is not None and action_features_tensor.device != config.device:
        action_features_tensor = action_features_tensor.to(config.device)
    action_features_tensor, principal_directions = process_feature_with_pca(action_features_tensor,
                                                                            q_value=500)

    predActionSeq = []  # * Initialize
    graphSeq_t = []
    # * Initialize environment.
    if INIT_DATAPOINT:
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=start_node)
        graphSeq_t.append(graphSeq[0])
    else:
        # * Use wold_home1 to test.
        # world_num = 1
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
        init_g = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            init_g = init_g.to(config.device)
        graphSeq_t.append(init_g)

    # * Generate actions and execute.
    selected_actions = []
    REPEAT_NUM = 0
    while True:
        # * Extract current task features.
        # if len(selected_actions) == 0:
        if len(selected_actions) <= (candidate_action_num - select_from_candidate):
            REPEAT_NUM += 1
            # # * Only generate candidate action pool once.
            # if REPEAT_NUM > 1:
            #     return 'Error', None, None, None
            selected_actions.clear()
            with torch.no_grad():
                _, current_task_feature = model_extract_feature(graphSeq_t[-1],
                                                                goal2vec)  # * [1, 512]
                current_task_feature, _ = process_feature_with_pca(current_task_feature,
                                                                   principal_directions)

            # * 01. Generation: Direct.
            # * Select actions.
            act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
            if act_generalized_inverse_mat.device != current_task_feature.device:
                act_generalized_inverse_mat = act_generalized_inverse_mat.to(current_task_feature.device)
            output = torch.matmul(current_task_feature, act_generalized_inverse_mat)  # * !!!
            output = output.squeeze(0)
            tpk = torch.topk(output, candidate_action_num)
            action_idx = tpk.indices
            action_idx = torch.sort(action_idx).values
            for idx in action_idx:
                selected_actions.append(action_names[idx])

        actions_prob = get_action_pred_with_model_actions(
            config, model_action, action_effect_features, graphSeq_t[-1], goal2vec
        )
        selected_actions_prob = []
        for act in selected_actions:
            idx = action_names.index(act)
            prob = actions_prob[0][idx]
            selected_actions_prob.append(prob)
        max_prob = max(selected_actions_prob)
        action_current_select = selected_actions[selected_actions_prob.index(max_prob)]
        selected_actions.remove(action_current_select)  # * Remove selected actions.

        # * !!!
        res, g, err = approx.execAction(config, action_current_select, config.embeddings)
        predActionSeq.append(action_current_select)
        if g is not None and config.device is not None:
            g = g.to(config.device)
        graphSeq_t.append(g)

        # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
        if res:
            return_data = None
            if (REPEAT_NUM == 1) or (len(predActionSeq) < len(actionSeq) + 5):
                return_data = [
                    goal_json, actionSeq, world_name, start_node, current_task_feature,
                    action_features_tensor, predActionSeq
                ]
            return 'Correct', len(actionSeq), len(predActionSeq), return_data
        elif err == '' and len(predActionSeq) > trajectory_length:
            return 'Incorrect', None, None, None
        elif err != '':
            return 'Error', None, None, None


def generate_qualitative_result(
        config,
        dataset,
        model_action,
        model_extract_feature,
        action_effect_features,
        TQDM=True,
        WITH_PCA=True,
        POOL_SET=None,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True,
        target_length=None
):
    """
    model_action: predict action
    model_extract_feature:
    """
    model_action.eval()
    model_extract_feature.eval()

    qualitative_data = []
    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        # * Get data.
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        if target_length is not None:
            if len(actionSeq) < target_length:
                continue

        if POOL_SET is None:
            multiscale_pool = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
        else:
            multiscale_pool = POOL_SET
        for (tmp_cdd_act_num, tmp_slt_act_num) in multiscale_pool:
            res, len_acts, len_pred_acts, return_data = qualitative_policy_with_feature_pca_order(
                config, model_action,
                model_extract_feature,
                data_item,
                _safe_numeric_suffix(world_name, default=0),
                action_effect_features,
                INIT_DATAPOINT,
                STATE_FORMAT_GOAL,
                candidate_action_num=tmp_cdd_act_num,
                select_from_candidate=tmp_slt_act_num
            )
            if res == 'Correct':
                if return_data is not None:
                    qualitative_data.append(return_data)
                break

    return qualitative_data


def check_policy_with_feature_pca_wo_order(
        config,
        model_action,
        model_extract_feature,
        data_item,
        world_num,
        action_effect_features,
        INIT_DATAPOINT,
        STATE_FORMAT_GOAL,
        candidate_action_num=20,
        select_from_candidate=10,
        trajectory_length=60
):
    model_action.eval()
    model_extract_feature.eval()
    # * Get data.
    if STATE_FORMAT_GOAL:  # * GraphDataset_State
        graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
    else:  # * GraphDataset
        graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    # * Convert to a tensor: each row is a [1, 512] action feature.
    # * squeeze(): [828, 1, 512] --> [828, 512], [action_num, feature_dim]
    action_features_tensor = torch.stack(action_features).squeeze(1)
    action_features_tensor, principal_directions = process_feature_with_pca(action_features_tensor,
                                                                            q_value=500)

    predActionSeq = []  # * Initialize
    graphSeq_t = []
    # * Initialize environment.
    if INIT_DATAPOINT:
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=start_node)
        graphSeq_t.append(graphSeq[0])
    else:
        # * Use wold_home1 to test.
        # world_num = 1
        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
        init_g = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            init_g = init_g.to(config.device)
        graphSeq_t.append(init_g)

    # * Generate actions and execute.
    selected_actions = []
    while True:
        # * Extract current task features.
        # if len(selected_actions) == 0:
        if len(selected_actions) <= (candidate_action_num - select_from_candidate):
            selected_actions.clear()
            with torch.no_grad():
                _, current_task_feature = model_extract_feature(graphSeq_t[-1],
                                                                goal2vec)  # * [1, 512]
                current_task_feature, _ = process_feature_with_pca(current_task_feature,
                                                                   principal_directions)

            # * 01. Generation: Direct.
            # * Select actions.
            act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
            output = torch.matmul(current_task_feature, act_generalized_inverse_mat)  # * !!!

            output = output.squeeze(0)
            tpk = torch.topk(output, candidate_action_num)
            action_idx = tpk.indices
            action_idx = torch.sort(action_idx).values
            for idx in action_idx:
                selected_actions.append(action_names[idx])
        action_current_select = selected_actions[-1]
        selected_actions.remove(action_current_select)  # * Remove selected actions.

        # * !!!
        res, g, err = approx.execAction(config, action_current_select, config.embeddings)
        predActionSeq.append(action_current_select)
        if g is not None and config.device is not None:
            g = g.to(config.device)
        graphSeq_t.append(g)

        # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
        if res:
            return 'Correct', len(actionSeq), len(predActionSeq)
        elif err == '' and len(predActionSeq) > trajectory_length:
            return 'Incorrect', None, None
        elif err != '':
            return 'Error', None, None


def test_policy_with_action_effect_features_wo_order(
        config,
        dataset,
        model_action,
        model_extract_feature,
        action_effect_features,
        multiscale_pool=None,
        TQDM=True,
        WITH_PCA=True,
        STATE_FORMAT_GOAL=True,
        INIT_DATAPOINT=True,
        IGNORE_ACTION_MODEL=False
):
    """
    model_action: predict action
    model_extract_feature:
    """
    model_action.eval()
    model_extract_feature.eval()
    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])

    data_num = {}
    data_correct_num = {}
    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item
        # * Store action length to dict.
        if str(len(actionSeq)) in data_num:
            data_num[str(len(actionSeq))] += 1
        else:
            data_num[str(len(actionSeq))] = 1
        if str(len(actionSeq)) not in data_correct_num:
            data_correct_num[str(len(actionSeq))] = 0
        world_num = _safe_numeric_suffix(world_name, default=0)
        plan_len = len(actionSeq)
        lenHuman.append(plan_len)

        if not IGNORE_ACTION_MODEL:
            res, len_acts, len_pred_acts = check_policy_with_model_actions(
                config, model_action, data_item, world_num, INIT_DATAPOINT, STATE_FORMAT_GOAL
            )
        else:
            res = 'False'

        if res == 'Correct':
            data_correct_num[str(len_acts)] += 1
            correct += 1
            lenModel.append(len_pred_acts)
        else:
            tmp_res = None
            # multiscale_pool = [(5, 3), (10, 3), (10, 5), (20, 5), (20, 10), (20, 15)]
            if multiscale_pool is None:
                multiscale_pool = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
            for (tmp_cdd_act_num, tmp_slt_act_num) in multiscale_pool:
                if WITH_PCA:
                    res, len_acts, len_pred_acts = check_policy_with_feature_pca_wo_order(
                        config,
                        model_action,
                        model_extract_feature,
                        data_item,
                        world_num,
                        action_effect_features,
                        INIT_DATAPOINT,
                        STATE_FORMAT_GOAL,
                        candidate_action_num=tmp_cdd_act_num,
                        select_from_candidate=tmp_slt_act_num
                    )
                else:
                    raise NotImplementedError
                    # res, len_acts, len_pred_acts = check_policy_with_feature(config, model_action,
                    #                                                          model_extract_feature,
                    #                                                          data_item,
                    #                                                          world_num,
                    #                                                          action_effect_features,
                    #                                                          INIT_DATAPOINT,
                    #                                                          STATE_FORMAT_GOAL,
                    #                                                          candidate_action_num=tmp_cdd_act_num,
                    #                                                          select_from_candidate=tmp_slt_act_num)

                tmp_res = res
                if res == 'Correct':
                    data_correct_num[str(len_acts)] += 1
                    correct += 1
                    lenModel.append(len_pred_acts)
                    break
            if tmp_res == 'Incorrect':
                incorrect += 1
            elif tmp_res == 'Error':
                error += 1

    # * Print results.
    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def scene_graph_goal_reached(config, current_graph, goal_graph, state_threshold=0.5):
    current_device = config.device if config.device is not None else 'cpu'
    current_state = current_graph.to(current_device).ndata['feat']
    target_state = goal_graph.ndata['feat']
    delta = torch.abs(target_state - current_state)
    return bool(torch.all(delta < state_threshold).item())


def plan_with_natural_language_instruction(
        config,
        model_action,
        model_extract_feature,
        action_effect_features,
        instruction,
        world_num,
        start_node=None,
        current_state_graph=None,
        current_scene_graph_json=None,
        qwen_model_name='qwen3-max',
        qwen_api_key=None,
        candidate_action_num=20,
        select_from_candidate=10,
        trajectory_length=60,
        with_pca=True,
        pca_q=500
):
    model_action.eval()
    model_extract_feature.eval()

    approx.initPolicy(
        config,
        config.domain,
        goal_json=None,
        world_num=world_num,
        SET_GAOL_JSON=False,
        INPUT_DATAPOINT=start_node
    )

    if current_state_graph is None:
        current_state_graph = approx.getInitializeDGLGraph(config)
    if config.device is not None:
        current_state_graph = current_state_graph.to(config.device)

    if current_scene_graph_json is None:
        current_scene_graph_json = get_current_scene_graph_json(config, state_name='Initialize')

    _, goal_scene_graph_json, goal_state_graph = translate_instruction_to_goal_state_graph(
        config,
        instruction=instruction,
        current_scene_graph_json=current_scene_graph_json,
        model_name=qwen_model_name,
        api_key=qwen_api_key
    )
    if config.device is not None:
        goal_state_graph = goal_state_graph.to(config.device)

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    action_features_tensor = torch.stack(action_features).squeeze(1)
    if config.device is not None:
        action_features_tensor = action_features_tensor.to(config.device)
    principal_directions = None
    if with_pca:
        action_features_tensor, principal_directions = process_feature_with_pca(
            action_features_tensor,
            q_value=pca_q
        )

    if scene_graph_goal_reached(config, current_state_graph, goal_state_graph):
        return {
            'status': 'Correct',
            'predicted_actions': [],
            'current_scene_graph_json': current_scene_graph_json,
            'goal_scene_graph_json': goal_scene_graph_json,
            'goal_state_graph': goal_state_graph,
            'final_state_graph': current_state_graph,
        }

    predActionSeq = []
    selected_actions = []

    while len(predActionSeq) < trajectory_length:
        if len(selected_actions) <= (candidate_action_num - select_from_candidate):
            selected_actions.clear()
            with torch.no_grad():
                _, current_task_feature = model_extract_feature(current_state_graph, goal_state_graph)
                if with_pca:
                    current_task_feature, _ = process_feature_with_pca(
                        current_task_feature,
                        principal_directions=principal_directions
                    )

            act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
            output = torch.matmul(current_task_feature, act_generalized_inverse_mat)
            output = output.squeeze(0)
            topk_actions = torch.topk(output, candidate_action_num)
            action_idx = torch.sort(topk_actions.indices).values
            for idx in action_idx:
                selected_actions.append(action_names[idx])

        actions_prob = get_action_pred_with_model_actions(
            config,
            model_action,
            action_effect_features,
            current_state_graph,
            goal_state_graph
        )

        selected_actions_prob = []
        for act in selected_actions:
            idx = action_names.index(act)
            selected_actions_prob.append(actions_prob[0][idx])

        max_prob = max(selected_actions_prob)
        action_current_select = selected_actions[selected_actions_prob.index(max_prob)]
        selected_actions.remove(action_current_select)

        _, next_graph, err = approx.execAction(config, action_current_select, config.embeddings)
        predActionSeq.append(action_current_select)

        if err != '':
            return {
                'status': 'Error',
                'predicted_actions': predActionSeq,
                'error': err,
                'current_scene_graph_json': current_scene_graph_json,
                'goal_scene_graph_json': goal_scene_graph_json,
                'goal_state_graph': goal_state_graph,
                'final_state_graph': current_state_graph,
            }

        if config.device is not None:
            next_graph = next_graph.to(config.device)
        current_state_graph = next_graph

        if scene_graph_goal_reached(config, current_state_graph, goal_state_graph):
            return {
                'status': 'Correct',
                'predicted_actions': predActionSeq,
                'current_scene_graph_json': current_scene_graph_json,
                'goal_scene_graph_json': goal_scene_graph_json,
                'goal_state_graph': goal_state_graph,
                'final_state_graph': current_state_graph,
            }

    return {
        'status': 'Incorrect',
        'predicted_actions': predActionSeq,
        'current_scene_graph_json': current_scene_graph_json,
        'goal_scene_graph_json': goal_scene_graph_json,
        'goal_state_graph': goal_state_graph,
        'final_state_graph': current_state_graph,
    }
