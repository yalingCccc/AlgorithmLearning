import torch
import math
from utils.utils import config, logging

SEED = 2021
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.logger

FEATURES = ['userid', 'feedid', 'hist_acts_seq', 'hist_acts_mask', 'no_act_seq', 'no_act_mask',
            'hist_read_comment_seq', 'hist_read_comment_mask', 'hist_like_seq', 'hist_like_mask',
            'hist_click_avatar_seq', 'hist_click_avatar_mask', 'hist_forward_seq', 'hist_forward_mask',
            'hist_favorite_seq', 'hist_favorite_mask', 'hist_comment_seq', 'hist_comment_mask', 'hist_follow_seq',
            'hist_follow_mask', 'finished_seq', 'finished_mask', 'unfinished_seq', 'unfinished_mask']
SEQFEATURES = ['hist_acts_seq', 'no_act_seq', 'hist_read_comment_seq', 'hist_like_seq', 'hist_click_avatar_seq',
               'hist_forward_seq', 'hist_favorite_seq', 'hist_comment_seq', 'hist_follow_seq', 'finished_seq',
               'unfinished_seq']
MASKFEATURES = ['hist_acts_mask', 'no_act_mask', 'hist_read_comment_mask', 'hist_like_mask', 'hist_click_avatar_mask',
                'hist_forward_mask', 'hist_favorite_mask', 'hist_comment_mask', 'hist_follow_mask', 'finished_mask',
                'unfinished_mask']
ACTIONS = ["read_comment", "like", "click_avatar", "forward"]
ALLACTIONS = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
WEIGHTS_MAP = {
    "read_comment": 4.0,  # 是否查看评论
    "like": 3.0,  # 是否点赞
    "click_avatar": 2.0,  # 是否点击头像
    "forward": 1.0,  # 是否转发
    "favorite": 1.0,  # 是否收藏
    "comment": 1.0,  # 是否发表评论
    "follow": 1.0  # 是否关注
}

arg_dict = {
    # 运行配置
    'mode': 'offline_train',
    'verbose_metric_step': 2000,
    'val_step': 1,

    # 路径配置
    'output_dir': './output/wechat',
    'model_dir': './output/wechat/models',
    'log_dir': './output/wechat/logs',
    'feed_info': 'data/wechat/wechat_algo_data1/feed_info.csv',
    'feed_embeddings': 'data/wechat/wechat_algo_data1/feed_embeddings.npz',
    'user_actions': 'data/wechat/wechat_algo_data1/user_action.csv',
    'train_dataset': 'data/wechat/train.csv',
    'test_dataset': 'data/wechat/test.csv',

    # 数据配置
    'train_day': 13,
    'test_day': 14,
    'max_hist_len': 50,
    'model_version': 4,
    'chunk_size': 51200,
    'val_size': 5120,
    'seq_feature_num': len(SEQFEATURES),
    'train_size': 566627,
    'vocab_size': {'feedid': 112872, 'userid': 260000},

    # 网络配置
    'batch_size': 256,
    'epoch_num': 5,
    'emb_size': 64,
    'feed_emb_fusion': 'dense',
    'emb_size_1': 64,  # feed_emb_fusion='dense'时自定义
    'input_dim': int(64 * (2 + len(SEQFEATURES))),  # 1664
    'l2': True,
    'optimizer': 'SGD',

    # MMoE
    'num_expert': 5,
    'num_task': len(ACTIONS),
    'expert_dims': [1024, 512, 256],
    'mmor_activation': 'relu',
    'mmoe_dropout': 0.2,
    'mmoe_batchnorm': False,
    'mmoe_mpl_dims': [512, 128, 1],
    'mmoe_mpl': True,

    # DIN
    'din_activation': 'Dice',
    'din_batchnorm': False,
    'din_dropout': 0,

    # 学习率衰减，warm up
    'base_lr': 1e-2,
    'final_lr': 1e-7,
    'warmup_begin_lr': 0,
    'warmup_mode': 'linear'  # 预热阶段：'linear'线性增加，'constant'固定值
}

# input dim
arg_dict['inter_output_dim'] = int((arg_dict['seq_feature_num'] + 2) * (arg_dict['seq_feature_num'] + 1) * 64 / 2)
arg_dict['mmoe_input_dim'] = int(arg_dict['input_dim'] + arg_dict['inter_output_dim'])

# warm up
arg_dict['max_warm_batch'] = math.ceil(arg_dict['train_size'] / arg_dict['batch_size'] * arg_dict['epoch_num'])
arg_dict['warmup_steps'] = math.ceil(arg_dict['max_warm_batch'] * 0.3)  # 30% warm up

args = config.get_parser(arg_dict)
