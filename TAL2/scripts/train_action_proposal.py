import os
import pickle
import time
import warnings

import colorama
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.config.config import init_args
from src.datasets.graph_dataset import APNExpandedGraphDataset, GraphDataset_State
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import get_model, load_model, save_model
from src.tal.utils_training import test_policy_graph_dataset
from src.utils.misc import setup_seed

colorama.init()
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


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


def _apn_loss(config, y_pred, y_true):
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
    return total_loss, {
        'loss_action': loss_action.detach().item(),
        'loss_obj1': loss_obj1.detach().item(),
        'loss_obj2_pos': loss_obj2_pos.detach().item(),
        'loss_obj2_neg': loss_obj2_neg.detach().item(),
        'loss_state': loss_state.detach().item(),
    }


def _pair_collate(batch):
    state_a = dgl.batch([item['state_a'] for item in batch])
    goal_state = dgl.batch([item['goal_state'] for item in batch])
    action_ab = torch.stack([item['action_ab'] for item in batch], dim=0)
    return {
        'state_a': state_a,
        'goal_state': goal_state,
        'action_ab': action_ab,
        'batch_size': len(batch),
    }


def _build_loader(dataset, batch_size, shuffle, num_workers):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': max(num_workers, 0),
        'collate_fn': _pair_collate,
        'pin_memory': torch.cuda.is_available(),
    }
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    return DataLoader(dataset, **loader_kwargs)


def _apply_argument_noise(batch, enabled):
    if not enabled:
        return batch['state_a'], batch['goal_state']

    state_a = batch['state_a'].to(batch['state_a'].device)
    goal_state = batch['goal_state'].to(batch['goal_state'].device)
    state_a.ndata = {key: value.clone() for key, value in state_a.ndata.items()}
    goal_state.ndata = {key: value.clone() for key, value in goal_state.ndata.items()}
    state_a.nodes = type(state_a.nodes)(state_a)
    goal_state.nodes = type(goal_state.nodes)(goal_state)

    noise = (torch.rand_like(state_a.ndata['feat']) * 2 - 1) * 0.2
    state_a.ndata['feat'] += noise
    return state_a, goal_state


def train_epoch(config, optimizer, scaler, loader, model, accum_steps=1, argument=False):
    model.train()

    metrics = {
        'total_loss': 0.0,
        'sample_count': 0,
        'loss_action': 0.0,
        'loss_obj1': 0.0,
        'loss_obj2_pos': 0.0,
        'loss_obj2_neg': 0.0,
        'loss_state': 0.0,
    }
    epoch_start = time.perf_counter()
    data_time = 0.0
    compute_time = 0.0
    optimizer.zero_grad(set_to_none=True)
    micro_step_count = 0

    iter_start = time.perf_counter()
    for batch in loader:
        data_time += time.perf_counter() - iter_start

        compute_start = time.perf_counter()
        state_a = _move_graph_to_device(batch['state_a'], config.device)
        goal_state = _move_graph_to_device(batch['goal_state'], config.device)
        action_ab = _move_tensor_to_device(batch['action_ab'], config.device)
        state_a, goal_state = _apply_argument_noise(
            {'state_a': state_a, 'goal_state': goal_state}, argument
        )

        with autocast(enabled=torch.cuda.is_available()):
            y_pred = model(state_a, goal_state)
            y_pred = _ensure_batch_dim(y_pred)
        with autocast(enabled=False):
            loss, loss_metrics = _apn_loss(config, y_pred.float(), action_ab.float())

        metrics['total_loss'] += loss.item() * batch['batch_size']
        metrics['sample_count'] += batch['batch_size']
        for key in ['loss_action', 'loss_obj1', 'loss_obj2_pos', 'loss_obj2_neg', 'loss_state']:
            metrics[key] += loss_metrics[key] * batch['batch_size']

        scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()
        micro_step_count += 1

        if micro_step_count % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        compute_time += time.perf_counter() - compute_start
        iter_start = time.perf_counter()

    if micro_step_count % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    total_time = time.perf_counter() - epoch_start
    metrics['average_loss'] = (
        metrics['total_loss'] / metrics['sample_count'] if metrics['sample_count'] > 0 else 0.0
    )
    for key in ['loss_action', 'loss_obj1', 'loss_obj2_pos', 'loss_obj2_neg', 'loss_state']:
        metrics[key] = metrics[key] / metrics['sample_count'] if metrics['sample_count'] > 0 else 0.0
    metrics['data_time'] = data_time
    metrics['compute_time'] = compute_time
    metrics['total_time'] = total_time
    metrics['samples_per_second'] = (
        metrics['sample_count'] / total_time if total_time > 0 else 0.0
    )
    return metrics


if __name__ == '__main__':
    rnd_seed = 1
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.policy_backend = 'symbolic'
    args.model_name = 'APN'
    args.num_epochs =   60 #####################
    config = EnvironmentConfig(args)

    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    train_sequence_dataset = GraphDataset_State(config, graphs_dir, train_data_path)

    val_data_path = './data/val_dataset.pkl'
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

    train_data_num = len(train_sequence_dataset)
    val_data_num = len(val_dataset)
    cprint('Train sequence num: {}'.format(train_data_num), 'green')
    cprint('Val data num: {}'.format(val_data_num), 'green')

    print('Expanding train sequences into APN pair samples...')
    expanded_train_dataset = APNExpandedGraphDataset(train_sequence_dataset)
    pair_dataset = expanded_train_dataset.get_pair_dataset()
    print('APN pair samples: {}'.format(len(pair_dataset)))

    num_workers = int(
        os.environ.get(
            'TAL_APN_NUM_WORKERS',
            '0' if os.name == 'nt' else str(min(8, os.cpu_count() or 1))
        )
    )
    batch_size = int(os.environ.get('TAL_APN_BATCH_SIZE', '64'))
    accum_steps = int(os.environ.get('TAL_APN_ACCUM_STEPS', '1'))
    print('APN batch size: {}'.format(batch_size))
    print('APN loader workers: {}'.format(num_workers))
    print('APN gradient accumulation steps: {}'.format(accum_steps))

    train_loader = _build_loader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    lr = 5e-4
    model, optimizer, epoch, accuracy_list = load_model(
        config, seqTool + model.name + '_Trained', model, lr=lr
    )
    model = model.to(config.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    test_frequency = 20
    if config.exec_type == 'train':
        print('Training ' + model.name + ' with ' + config.embedding)
        for epoch_num in range(epoch + 1, config.NUM_EPOCHS + 1):
            if epoch_num > 300:
                test_frequency = 10
            scheduler.step(epoch=epoch_num)
            print('EPOCH {}, lr: {}'.format(epoch_num, optimizer.param_groups[0]['lr']))

            metrics = train_epoch(
                config,
                optimizer,
                scaler,
                train_loader,
                model,
                accum_steps=accum_steps,
                argument=False,
            )

            cprint('Time per epoch: {}'.format(metrics['total_time']), 'green')
            print('Loss sum: {}'.format(metrics['total_loss']))
            print('Loss avg: {}'.format(metrics['average_loss']))
            print(
                'Loss parts: action={:.4f} obj1={:.4f} obj2_pos={:.4f} obj2_neg={:.4f} state={:.4f}'.format(
                    metrics['loss_action'],
                    metrics['loss_obj1'],
                    metrics['loss_obj2_pos'],
                    metrics['loss_obj2_neg'],
                    metrics['loss_state'],
                )
            )
            print(
                'Epoch timing: data={:.2f}s compute={:.2f}s total={:.2f}s'.format(
                    metrics['data_time'], metrics['compute_time'], metrics['total_time']
                )
            )
            print('Throughput: {:.2f} samples/s'.format(metrics['samples_per_second']))

            c, i, e = 0, 0, 0
            if (metrics['average_loss'] < 100) and ((epoch_num + 1) % test_frequency == 0):
                cprint('Val data policy test.', 'green')
                c, i, e, _, _ = test_policy_graph_dataset(
                    config, val_dataset, model, config.num_objects, TQDM=False
                )
            accuracy_list.append((0, 0, metrics['average_loss'], c, i, e))

            target_path = f"{config.MODEL_SAVE_PATH}/{seqTool}{model.name}_Trained.ckpt"
            save_model(config, model, optimizer, epoch_num, accuracy_list, file_path=target_path)

        print('The maximum accuracy on test set is ', str(max(accuracy_list)), ' at epoch ',
              accuracy_list.index(max(accuracy_list)))

        policy_acc = [i[3] for i in accuracy_list]
        print('The maximum policy on test set is ', str(max(policy_acc)), ' at epoch ',
              policy_acc.index(max(policy_acc)))

        results_save_path = './' + config.MODEL_SAVE_PATH + 'results.pkl'
        with open(results_save_path, 'wb') as f:
            pickle.dump(accuracy_list, f)
