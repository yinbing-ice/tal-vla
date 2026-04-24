import os
import pickle
import re
import time
import warnings

import colorama
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from torch import optim
from torch.utils.data import DataLoader

from src.config.config import init_args
from src.datasets.graph_dataset import AFEExpandedGraphDataset, GraphDataset_State
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import accuracy_score_feature_extractor
from src.tal.utils_training import get_model, load_model, save_model
from src.utils.misc import setup_seed

colorama.init()
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


def resolve_resume_checkpoint(config, model_name, seq_prefix=''):
    """Prefer an explicit stable checkpoint, otherwise resume from the latest epoch file."""
    model_dir = config.MODEL_SAVE_PATH
    stable_ckpt = os.path.join(model_dir, f'{seq_prefix}{model_name}_Trained.ckpt')
    if os.path.exists(stable_ckpt):
        return stable_ckpt

    pattern = re.compile(rf'^{re.escape(seq_prefix + model_name)}_(\d+)\.ckpt$')
    latest_epoch = -1
    latest_ckpt = None
    if not os.path.isdir(model_dir):
        return None

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_ckpt = os.path.join(model_dir, filename)
    return latest_ckpt


def _move_graph_to_device(graph, device):
    return graph.to(device) if device is not None else graph


def _move_tensor_to_device(tensor, device):
    return tensor.to(device, non_blocking=True) if device is not None else tensor


def _ensure_batch_dim(tensor):
    if torch.is_tensor(tensor) and tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _zero_like_loss(tensor):
    return tensor.sum() * 0.0


def _categorical_nll_from_probs(pred_probs, target_one_hot):
    mask = target_one_hot.sum(dim=1) > 0
    if not torch.any(mask):
        return _zero_like_loss(pred_probs)
    pred_probs = pred_probs[mask].clamp_min(1e-8)
    target_one_hot = target_one_hot[mask]
    return -(target_one_hot * pred_probs.log()).sum(dim=1).mean()


def _split_action_tensor(config, tensor):
    num_actions = len(config.possibleActions)
    num_objects = config.num_objects
    idx0 = num_actions
    idx1 = idx0 + num_objects
    idx2 = idx1 + num_objects
    return tensor[:, :idx0], tensor[:, idx0:idx1], tensor[:, idx1:idx2], tensor[:, idx2:]


def _structured_action_loss(config, y_pred, y_true):
    pred_action, pred_obj1, pred_obj2, pred_state = _split_action_tensor(config, y_pred)
    true_action, true_obj1, true_obj2, true_state = _split_action_tensor(config, y_true)

    loss_action = _categorical_nll_from_probs(pred_action, true_action)
    loss_obj1 = _categorical_nll_from_probs(pred_obj1, true_obj1)

    two_arg_mask = true_obj2.sum(dim=1) > 0
    if torch.any(two_arg_mask):
        loss_obj2_pos = _categorical_nll_from_probs(pred_obj2[two_arg_mask], true_obj2[two_arg_mask])
    else:
        loss_obj2_pos = _zero_like_loss(pred_obj2)

    if torch.any(~two_arg_mask):
        loss_obj2_neg = F.binary_cross_entropy(pred_obj2[~two_arg_mask], true_obj2[~two_arg_mask])
    else:
        loss_obj2_neg = _zero_like_loss(pred_obj2)

    loss_state = F.binary_cross_entropy(pred_state, true_state)
    total_loss = loss_action + loss_obj1 + loss_obj2_pos + 0.1 * loss_obj2_neg + 0.1 * loss_state
    metrics = {
        'action_name_loss': loss_action.detach().item(),
        'arg1_loss': loss_obj1.detach().item(),
        'arg2_pos_loss': loss_obj2_pos.detach().item(),
        'arg2_neg_loss': loss_obj2_neg.detach().item(),
        'state_loss': loss_state.detach().item(),
    }
    return total_loss, metrics


def _pair_collate(batch):
    state_a = dgl.batch([item['state_a'] for item in batch])
    state_b = dgl.batch([item['state_b'] for item in batch])
    action_ab = torch.stack([item['action_ab'] for item in batch], dim=0)
    return {
        'state_a': state_a,
        'state_b': state_b,
        'action_ab': action_ab,
        'batch_size': len(batch),
    }


def _triplet_collate(batch):
    state_a = dgl.batch([item['state_a'] for item in batch])
    state_b = dgl.batch([item['state_b'] for item in batch])
    state_c = dgl.batch([item['state_c'] for item in batch])
    action_ab = torch.stack([item['action_ab'] for item in batch], dim=0)
    action_bc = torch.stack([item['action_bc'] for item in batch], dim=0)
    return {
        'state_a': state_a,
        'state_b': state_b,
        'state_c': state_c,
        'action_ab': action_ab,
        'action_bc': action_bc,
        'batch_size': len(batch),
    }


def _build_loader(dataset, batch_size, shuffle, num_workers, collate_fn):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': max(num_workers, 0),
        'collate_fn': collate_fn,
        'pin_memory': torch.cuda.is_available(),
    }
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    return DataLoader(dataset, **loader_kwargs)


def _rowwise_action_similarity(action_ab, action_bc):
    return torch.where(
        torch.all(torch.isclose(action_ab, action_bc), dim=1),
        torch.ones(action_ab.shape[0], device=action_ab.device),
        -torch.ones(action_ab.shape[0], device=action_ab.device),
    )


def train_epoch(config, optimizer, model, pair_loader, triplet_loader, accum_steps=1):
    model.train()
    criterion_diff_actions = nn.CosineEmbeddingLoss(margin=0.2)
    criterion_feature = nn.MSELoss()

    metrics = {
        'total_loss': 0.0,
        'action_loss': 0.0,
        'feature_loss': 0.0,
        'diff_action_loss': 0.0,
        'pair_samples': 0,
        'triplet_samples': 0,
        'action_name_loss': 0.0,
        'arg1_loss': 0.0,
        'arg2_pos_loss': 0.0,
        'arg2_neg_loss': 0.0,
        'state_loss': 0.0,
    }

    epoch_start = time.perf_counter()
    data_time = 0.0
    compute_time = 0.0
    optimizer.zero_grad(set_to_none=True)
    micro_step_count = 0

    triplet_iter_start = time.perf_counter()
    for batch in triplet_loader:
        data_time += time.perf_counter() - triplet_iter_start

        compute_start = time.perf_counter()
        state_a = _move_graph_to_device(batch['state_a'], config.device)
        state_b = _move_graph_to_device(batch['state_b'], config.device)
        state_c = _move_graph_to_device(batch['state_c'], config.device)
        action_ab = _move_tensor_to_device(batch['action_ab'], config.device)
        action_bc = _move_tensor_to_device(batch['action_bc'], config.device)

        y_pred_ab, delta_ab = model(state_a, state_b)
        y_pred_bc, delta_bc = model(state_b, state_c)
        _, delta_ac = model(state_a, state_c)
        y_pred_ab = _ensure_batch_dim(y_pred_ab)
        y_pred_bc = _ensure_batch_dim(y_pred_bc)
        delta_ab = _ensure_batch_dim(delta_ab)
        delta_bc = _ensure_batch_dim(delta_bc)
        delta_ac = _ensure_batch_dim(delta_ac)

        loss_action_1, loss_metrics_ab = _structured_action_loss(config, y_pred_ab, action_ab)
        loss_action_2, loss_metrics_bc = _structured_action_loss(config, y_pred_bc, action_bc)
        loss_feature = criterion_feature(delta_ac, delta_ab + delta_bc)
        y_cosine_embed = _rowwise_action_similarity(action_ab, action_bc)
        loss_diff_actions = criterion_diff_actions(delta_ab, delta_bc, y_cosine_embed)
        loss = loss_action_1 + loss_action_2 + loss_feature + loss_diff_actions

        (loss / accum_steps).backward()
        micro_step_count += 1
        if micro_step_count % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics['total_loss'] += loss.item()
        metrics['action_loss'] += (loss_action_1.item() + loss_action_2.item())
        metrics['feature_loss'] += loss_feature.item()
        metrics['diff_action_loss'] += loss_diff_actions.item()
        metrics['triplet_samples'] += batch['batch_size']
        for key in ['action_name_loss', 'arg1_loss', 'arg2_pos_loss', 'arg2_neg_loss', 'state_loss']:
            metrics[key] += (loss_metrics_ab[key] + loss_metrics_bc[key]) * batch['batch_size']
        compute_time += time.perf_counter() - compute_start
        triplet_iter_start = time.perf_counter()

    pair_iter_start = time.perf_counter()
    for batch in pair_loader:
        data_time += time.perf_counter() - pair_iter_start

        compute_start = time.perf_counter()
        state_a = _move_graph_to_device(batch['state_a'], config.device)
        state_b = _move_graph_to_device(batch['state_b'], config.device)
        action_ab = _move_tensor_to_device(batch['action_ab'], config.device)

        y_pred_ab, _ = model(state_a, state_b)
        y_pred_ab = _ensure_batch_dim(y_pred_ab)
        loss, loss_metrics_ab = _structured_action_loss(config, y_pred_ab, action_ab)

        (loss / accum_steps).backward()
        micro_step_count += 1
        if micro_step_count % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics['total_loss'] += loss.item()
        metrics['action_loss'] += loss.item()
        metrics['pair_samples'] += batch['batch_size']
        for key in ['action_name_loss', 'arg1_loss', 'arg2_pos_loss', 'arg2_neg_loss', 'state_loss']:
            metrics[key] += loss_metrics_ab[key] * batch['batch_size']
        compute_time += time.perf_counter() - compute_start
        pair_iter_start = time.perf_counter()

    if micro_step_count % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    total_time = time.perf_counter() - epoch_start
    processed_samples = metrics['pair_samples'] + metrics['triplet_samples']
    metrics['data_time'] = data_time
    metrics['compute_time'] = compute_time
    metrics['total_time'] = total_time
    metrics['samples_per_second'] = (
        processed_samples / total_time if total_time > 0 else 0.0
    )
    metrics['triplets_per_second'] = (
        metrics['triplet_samples'] / total_time if total_time > 0 else 0.0
    )
    processed_action_losses = metrics['pair_samples'] + metrics['triplet_samples'] * 2
    for key in ['action_name_loss', 'arg1_loss', 'arg2_pos_loss', 'arg2_neg_loss', 'state_loss']:
        metrics[key] = metrics[key] / processed_action_losses if processed_action_losses > 0 else 0.0
    return metrics


if __name__ == '__main__':
    rnd_seed = 42
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    args.num_epochs = 1045  ##修改轮数
    config = EnvironmentConfig(args)

    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    train_sequence_dataset = GraphDataset_State(config, graphs_dir, train_data_path)

    val_data_path = './data/val_dataset.pkl'
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)


    train_data_num = len(train_sequence_dataset)
    val_data_num = len(val_dataset)

    cprint('--' * 20, 'green')
    print('Train sequence num: {}'.format(train_data_num))
    print('Val data num: {}'.format(val_data_num))
    cprint('--' * 20, 'green')

    print('Expanding train sequences into AFE pair/triplet samples...')
    expanded_train_dataset = AFEExpandedGraphDataset(train_sequence_dataset)
    pair_dataset = expanded_train_dataset.get_pair_dataset()
    triplet_dataset = expanded_train_dataset.get_triplet_dataset()
    print('AFE pair samples: {}'.format(len(pair_dataset)))
    print('AFE triplet samples: {}'.format(len(triplet_dataset)))

    num_workers = int(
        os.environ.get(
            'TAL_AFE_NUM_WORKERS',
            '0' if os.name == 'nt' else str(min(8, os.cpu_count() or 1))
        )
    )
    batch_size = int(os.environ.get('TAL_AFE_BATCH_SIZE', '32'))
    accum_steps = int(os.environ.get('TAL_AFE_ACCUM_STEPS', '1'))
    print('AFE batch size: {}'.format(batch_size))
    print('AFE loader workers: {}'.format(num_workers))
    print('AFE gradient accumulation steps: {}'.format(accum_steps))

    pair_loader = _build_loader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_pair_collate,
    )
    triplet_loader = _build_loader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_triplet_collate,
    )

    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''

    lr = 1e-4
    resume_ckpt = resolve_resume_checkpoint(config, model.name, seq_prefix=seqTool)
    if resume_ckpt is not None:
        print('Resuming AFE training from latest checkpoint: {}'.format(resume_ckpt))
    else:
        print('No existing AFE checkpoint found. Training will start from scratch.')
    model, optimizer, epoch, accuracy_list = load_model(
        config,
        seqTool + model.name + '_Trained',
        model,
        file_path=resume_ckpt,
        lr=lr
    )
    model = model.to(config.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    target_path = f"{config.MODEL_SAVE_PATH}/{seqTool}{model.name}_Trained.ckpt"

    test_frequency = 5
    if config.exec_type == 'train':
        print('Training ' + model.name + ' with ' + config.embedding)
        for epoch_num in range(epoch + 1, config.NUM_EPOCHS):
            scheduler.step(epoch=epoch_num)
            print('EPOCH {}, lr: {}'.format(epoch_num, optimizer.param_groups[0]['lr']))
            metrics = train_epoch(
                config,
                optimizer,
                model,
                pair_loader,
                triplet_loader,
                accum_steps=accum_steps,
            )
            print('Total loss: {}'.format(metrics['total_loss']))
            print('Action loss: {}'.format(metrics['action_loss']))
            print('Feature loss: {}'.format(metrics['feature_loss']))
            print('Diff action feature loss: {}'.format(metrics['diff_action_loss']))
            print(
                'Action parts: name={:.4f} arg1={:.4f} arg2_pos={:.4f} arg2_neg={:.4f} state={:.4f}'.format(
                    metrics['action_name_loss'],
                    metrics['arg1_loss'],
                    metrics['arg2_pos_loss'],
                    metrics['arg2_neg_loss'],
                    metrics['state_loss'],
                )
            )
            print(
                'Epoch timing: data={:.2f}s compute={:.2f}s total={:.2f}s'.format(
                    metrics['data_time'], metrics['compute_time'], metrics['total_time']
                )
            )
            print(
                'Throughput: {:.2f} samples/s, {:.2f} triplets/s'.format(
                    metrics['samples_per_second'], metrics['triplets_per_second']
                )
            )

            t1, t2, t3 = 0, 0, 0
            if (
                (epoch_num > 50)
                and (metrics['total_loss'] < 50)
                and (epoch_num + 1) % test_frequency == 0
            ):
                t1 = accuracy_score_feature_extractor(
                    config, train_sequence_dataset, model, config.num_objects, TQDM=False
                )
                print('Train Accuracy (action): {}'.format(t1))
                t2 = accuracy_score_feature_extractor(
                    config, val_dataset, model, config.num_objects, TQDM=False
                )
                print('Val Accuracy (action): {}'.format(t2))

            accuracy_list.append(
                (
                    t1,
                    t2,
                    t3,
                    metrics['total_loss'],
                    metrics['action_loss'],
                    metrics['feature_loss'],
                )
            )

            save_model(config, model, optimizer, epoch_num, accuracy_list, file_path=target_path)

        train_acc = [i[0] for i in accuracy_list]
        val_acc = [i[1] for i in accuracy_list]
        print('The maximum acc on train set is ', str(max(train_acc)), ' at epoch ',
              train_acc.index(max(train_acc)))
        print('The maximum acc on val set is ', str(max(val_acc)), ' at epoch ',
              val_acc.index(max(val_acc)))

        results_save_path = './' + config.MODEL_SAVE_PATH + 'feature_extractor_results.pkl'
        with open(results_save_path, 'wb') as f:
            pickle.dump(accuracy_list, f)
