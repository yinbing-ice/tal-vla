import pickle
import traceback
import torch
import colorama
import warnings
from torch.utils.data import Subset
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import get_model, load_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.envs import approx
from src.datasets.graph_dataset import GraphDataset_State
from src.tal.utils_planning import test_policy_with_action_effect_features

colorama.init()
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = init_args()
    # 强制指定设备和模型名
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    config = EnvironmentConfig(args)
    
    try:
        # * ------------------------------------------------------------------------------------------
        # * 1. 加载数据
        # * ------------------------------------------------------------------------------------------
        graphs_dir = './data/home/'
        test_data_path = './data/test_dataset.pkl'
        
        print(f">>> Loading full test dataset from {test_data_path}...")
        full_test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)

        # 🟢 核心修改：筛选长度
        print(">>> Filtering samples with action length 8 and 9...")
        filtered_indices = []
        for i in range(len(full_test_dataset)):
            # full_test_dataset[i] 返回的是 (graphSeq, goal2vec, ..., actionSeq, ...)
            # 索引 3 通常是 actionSeq (动作序列)
            action_len = len(full_test_dataset[i][3]) 
            if action_len in [10]: # 🟢 修改长度
                filtered_indices.append(i)
        
        # 只取前 100 个符合条件的样本
        selected_indices = filtered_indices[:100]
        test_dataset = Subset(full_test_dataset, selected_indices)
        
        print(f'>>> Target lengths (4,5) found: {len(filtered_indices)} samples.') # 🟢 修改
        print(f'>>> Final test size restricted to: {len(test_dataset)} samples.')

        # * ------------------------------------------------------------------------------------------
        # * 2. 加载模型 AFE & APN
        # * ------------------------------------------------------------------------------------------
        # AFE
        model_action_effect = get_model(config, config.model_name, config.features_dim, config.num_objects)
        seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
        model_action_effect, _, epoch_afe, _ = load_model(config, seqTool + model_action_effect.name + '_Trained', model_action_effect)
        model_action_effect = model_action_effect.to(config.device)
        print(f'Model: AFE | epoch: {epoch_afe}')

        # APN
        model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
        model_action, _, epoch_apn, _ = load_model(config, seqTool + model_action.name + '_Trained', model_action)
        model_action = model_action.to(config.device)
        print(f'Model: APN | epoch: {epoch_apn}')

        # 加载特征库
        features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
        with open(features_save_path, 'rb') as f:
            action_effect_features = pickle.load(f)
        print(f'Loaded {len(action_effect_features["names"])} action effect features.')

        # * ------------------------------------------------------------------------------------------
        # * 3. 执行 4-5 步长测试
        # * ------------------------------------------------------------------------------------------
        MCAS = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
        PCA_FLAG = True
        seed = 42
        setup_seed(seed=seed)

        print('\n' + '='*50)
        print(f'STARTING EVALUATION: Action Length 4-5 Only') # 🟢 修改
        print(f'Random seed: {seed} | Samples: {len(test_dataset)}')
        print('='*50 + '\n')

        test_policy_with_action_effect_features(config,
                                                test_dataset,
                                                model_action,
                                                model_action_effect,
                                                action_effect_features,
                                                multiscale_pool=MCAS,
                                                TQDM=True,
                                                ONLY_ACTION_MODEL=False,
                                                ONLY_FEATURE_MODEL=False,
                                                WITH_PCA=PCA_FLAG,
                                                STATE_FORMAT_GOAL=True,
                                                INIT_DATAPOINT=True,
                                                IGNORE_ACTION_MODEL=True,
                                                MAX_SAMPLES=-1, # 已经通过 Subset 限制了，这里设为 -1
                                                PRINT_SAMPLE_INFO=False, # 🟢开启实时打印，每步都看得到True
                                                PROGRESS_DESC='TAL Test (10 steps)') # 🟢 修改

    except Exception:
        print('\n[test_policy_tal] Error occurred:')
        traceback.print_exc()
        raise
    finally:
        approx.close_backend()

# import pickle
# import traceback
# import torch
# import colorama
# import warnings
# from src.utils.misc import setup_seed
# from src.config.config import init_args
# from src.tal.utils_training import get_model, load_model
# from src.envs.CONSTANTS import EnvironmentConfig
# from src.envs import approx
# from src.datasets.graph_dataset import GraphDataset_State
# from src.tal.utils_planning import test_policy_with_action_effect_features

# colorama.init()
# warnings.filterwarnings('ignore')

# if __name__ == '__main__':
#     args = init_args()
#     args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     args.model_name = 'AFE'
#     config = EnvironmentConfig(args)
#     try:
#         # * ------------------------------------------------------------------------------------------
#         # * Load data.
#         graphs_dir = './data/home/'
#         train_data_path = './data/train_dataset.pkl'
#         train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
#         #🟢 新增：强制只取前 100 个样本
#         import torch.utils.data as data
#         train_dataset = data.Subset(train_dataset, range(min(100, len(train_dataset))))
#         print('Force Limit: Train data num changed to: {}'.format(len(train_dataset)))

#         print('Train data num: {}'.format(len(train_dataset)))

#         # val_data_path = './data/val_dataset.pkl'
#         # val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

#         test_data_path = './data/test_dataset.pkl'
#         test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)
#         # 🟢 新增：强制只取前 100 个样本
#         test_dataset = data.Subset(test_dataset, range(min(100, len(test_dataset))))
#         print('Force Limit: Test data num changed to: {}'.format(len(test_dataset)))
#         print('Test data num: {}'.format(len(test_dataset)))
#         if args.max_samples >= 0:
#             print('Debug max_samples: {}'.format(args.max_samples))
#         print('Debug print_sample_info: {}'.format(args.print_sample_info))

#         # * ------------------------------------------------------------------------------------------
#         # * Create model and load parameters.
#         # * 01.Action Feature Extractor: AFE
#         model_action_effect = get_model(config, config.model_name, config.features_dim,
#                                         config.num_objects)
#         seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
#         model_action_effect, optimizer, epoch, accuracy_list = load_model(
#             config,
#             seqTool + model_action_effect.name + '_Trained',
#             model_action_effect
#         )
#         print('Model: {} | epoch: {}'.format(model_action_effect.name, epoch))
#         model_action_effect = model_action_effect.to(config.device)

#         # * 02.Action Proposal Network: APN
#         model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
#         seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
#         model_action, optimizer, epoch, accuracy_list = load_model(
#             config,
#             seqTool + model_action.name + '_Trained',
#             model_action
#         )
#         print('Model: APN | epoch: {}'.format(epoch))
#         model_action = model_action.to(config.device)
#         print('Policy backend: {}'.format(config.policy_backend))

#         # * ------------------------------------------------------------------------------------------
#         # * Load features.
#         features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
#         with open(features_save_path, 'rb') as f:
#             action_effect_features = pickle.load(f)
#         print('Action effect features num: {}'.format(len(action_effect_features['names'])))

#         # * ------------------------------------------------------------------------------------------
#         # * Policy test.
#         seeds = [42]
#         MCAS = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
#         PCA_FLAG = True

#         print('\n\n')
#         print('//' * 20)
#         print('Candidate Action Pool (w/ PCA, Early Stopping)')
#         print('pool_sets: {}'.format(MCAS))
#         print('PCA_FLAG: {}'.format(PCA_FLAG))
#         print('//' * 20)

#         print('\n\n-----------------------------------------')
#         print('Test on training set.')
#         print('-----------------------------------------')
#         for seed in seeds:
#             setup_seed(seed=seed)
#             print('Random seed: {}'.format(seed))
#             print('Evaluating {} samples from train_dataset...'.format(len(train_dataset)))
#             test_policy_with_action_effect_features(config,
#                                                     train_dataset,
#                                                     model_action,
#                                                     model_action_effect,
#                                                     action_effect_features,
#                                                     multiscale_pool=MCAS,
#                                                     TQDM=True,
#                                                     ONLY_ACTION_MODEL=False,
#                                                     ONLY_FEATURE_MODEL=False,
#                                                     WITH_PCA=PCA_FLAG,
#                                                     STATE_FORMAT_GOAL=True,
#                                                     INIT_DATAPOINT=True,
#                                                     IGNORE_ACTION_MODEL=True,
#                                                     MAX_SAMPLES=args.max_samples,
#                                                     PRINT_SAMPLE_INFO=args.print_sample_info,
#                                                     PROGRESS_DESC='TAL Train')

#         # print('\n\n-----------------------------------------')
#         # print('Test on val set.')
#         # print('-----------------------------------------')
#         # for seed in seeds:
#         #     setup_seed(seed=seed)
#         #     print('Random seed: {}'.format(seed))
#         #     test_policy_with_action_effect_features(config,
#         #                                             val_dataset,
#         #                                             model_action,
#         #                                             model_action_effect,
#         #                                             action_effect_features,
#         #                                             multiscale_pool=MCAS,
#         #                                             TQDM=True,
#         #                                             ONLY_ACTION_MODEL=False,
#         #                                             ONLY_FEATURE_MODEL=False,
#         #                                             WITH_PCA=PCA_FLAG,
#         #                                             STATE_FORMAT_GOAL=True,
#         #                                             INIT_DATAPOINT=True,
#         #                                             IGNORE_ACTION_MODEL=True)

#         # print('\n\n-----------------------------------------')
#         # print('Test on test set.')
#         # print('-----------------------------------------')
#         # for seed in seeds:
#         #     setup_seed(seed=seed)
#         #     print('Random seed: {}'.format(seed))
#         #     print('Evaluating {} samples from test_dataset...'.format(len(test_dataset)))
#         #     test_policy_with_action_effect_features(config,
#         #                                             test_dataset,
#         #                                             model_action,
#         #                                             model_action_effect,
#         #                                             action_effect_features,
#         #                                             multiscale_pool=MCAS,
#         #                                             TQDM=True,
#         #                                             ONLY_ACTION_MODEL=False,
#         #                                             ONLY_FEATURE_MODEL=False,
#         #                                             WITH_PCA=PCA_FLAG,
#         #                                             STATE_FORMAT_GOAL=True,
#         #                                             INIT_DATAPOINT=True,
#         #                                             IGNORE_ACTION_MODEL=True,
#         #                                             MAX_SAMPLES=args.max_samples,
#         #                                             PRINT_SAMPLE_INFO=args.print_sample_info,
#         #                                             PROGRESS_DESC='TAL Test')
#     except Exception:
#         print('\n[test_policy_tal] Unhandled exception during policy evaluation:')
#         traceback.print_exc()
#         raise
#     finally:
#         approx.close_backend()
