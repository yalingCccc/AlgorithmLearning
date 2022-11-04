import numpy as np
import pandas as pd
import torch
import math
import os
import gc
from utils.utils.pickle_pro import pickle_process
from utils.utils import config, logging

SEED = 2021
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.logger

# 特征
BASE_FEATURES = ['userid', 'feedid']
CONTEXT_FEATURES = ['device']
SEQ_FEATURES = ['hist_acts_seq', 'no_act_seq', 'hist_read_comment_seq', 'hist_like_seq', 'hist_click_avatar_seq',
                'hist_forward_seq', 'hist_favorite_seq', 'hist_comment_seq', 'hist_follow_seq', 'finished_seq',
                'unfinished_seq']
MASK_FEATURES = ['hist_acts_mask', 'no_act_mask', 'hist_read_comment_mask', 'hist_like_mask', 'hist_click_avatar_mask',
                 'hist_forward_mask', 'hist_favorite_mask', 'hist_comment_mask', 'hist_follow_mask', 'finished_mask',
                 'unfinished_mask']
SIDE_INFORMATIONS = []
# SIDE_INFORMATIONS = ['authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'manual_keyword_list',
#                      'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']
FEATURES = BASE_FEATURES + CONTEXT_FEATURES + SEQ_FEATURES

# 标签
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

# 默认配置
init_params = {
    # 运行配置
    'mode': 'offline_train',
    'verbose_metric_step': 1,
    'val_step': 1,

    # 路径配置
    # output
    'output_dir': 'output/wechat',
    'model_dir': 'output/wechat/models',
    'log_dir': 'output/wechat/logs',
    # data process 数据路径
    'feed_info': 'data/wechat/wechat_algo_data1/feed_info.csv',
    'feed_embeddings': 'data/wechat/wechat_algo_data1/feed_embeddings.csv',
    'user_actions': 'data/wechat/wechat_algo_data1/user_action.csv',
    # model train 读入数据路径
    'train_dataset': 'output/wechat/train.csv',
    'test_dataset': 'output/wechat/test.csv',
    'pretrain_embeddings': 'output/wechat/pretrain_emb.pkl',
    'vocab_dict_path': 'output/wechat/vocab_size.pkl',
    'feed_info_pro': 'output/wechat/side_info.pkl',

    # 数据配置
    'train_day': 13,
    'test_day': 14,
    'max_hist_len': 50,
    'model_version': 4,
    'chunk_size': 51200,
    'val_size': 5120,
    'seq_feature_num': len(SEQ_FEATURES),
    'train_size': 566627,
    'feed_pretrained': True,
    'user_pretrained': True,
    'vocab_size': {'feedid': 112872,
                   'userid': 250236,
                   'authorid': 18788,
                   'videoplayseconds': 3,
                   'bgm_song_id': 25158,
                   'bgm_singer_id': 17499,
                   'manual_keyword': 4,
                   'machine_keyword': 3,
                   'manual_tag': 2,
                   'machine_tag': 1,
                   'device': 5
                   },

    # 网络配置
    'batch_size': 128,
    'epoch_num': 5,
    'emb_size': 64,
    'feed_emb_fusion': 'dense',
    'l2': True,
    'optimizer': 'Adam',

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
    'base_lr': 1e-4,
    'final_lr': 1e-6,
    'warmup_begin_lr': 1e-6,
    'warmup_mode': 'linear'  # 预热阶段：'linear'线性增加，'constant'固定值
}

hparams = vars(config.get_parser(init_params))

# input message
emb_num = len(BASE_FEATURES) + len(SEQ_FEATURES)
if len(CONTEXT_FEATURES) > 0:
    emb_num += 1
hparams['emb_num'] = emb_num

hparams['input_dim'] = hparams['emb_size'] * hparams['emb_num']
hparams['inter_output_dim'] = hparams['emb_num'] * (hparams['emb_num'] - 1) / 2 * hparams['emb_size']
hparams['mmoe_input_dim'] = int(hparams['input_dim'] + hparams['inter_output_dim'])

# warm up
if os.path.exists(hparams['train_dataset']):
    train_dataset = pd.read_csv(hparams['train_dataset'])
    hparams['train_size'] = train_dataset.shape[0]
    del train_dataset
    gc.collect()
hparams['max_warm_batch'] = math.ceil(hparams['train_size'] / hparams['batch_size'] * hparams['epoch_num'])
hparams['warmup_steps'] = math.ceil(hparams['max_warm_batch'] * 0.3)  # 30% warm up
if os.path.exists(hparams['vocab_dict_path']):
    vocab_size = pickle_process('load', hparams['vocab_dict_path'])
    hparams['vocab_size'] = {key: int(value) for key, value in vocab_size.items()}
f = open(os.path.join(hparams['output_dir'], 'hparameters.txt'), 'w')
f.write(str(hparams))
f.close()
