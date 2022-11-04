import torch.cuda
from absl import app
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import DLRM
from config import *
from metrics import score
from utils.lr.warmup_scheduler import CosineScheduler
from dataset import DataReader
from utils.utils.optimizer import get_optimizer
# from functools import partial
import os

writer = SummaryWriter(log_dir=hparams['log_dir'])
np.random.seed(SEED)  # 同时也要设置numpy
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_loss(y_hat, target, loss_fn):
    loss1 = loss_fn(y_hat[0], target[0].view(-1, 1).float()) * 4
    loss2 = loss_fn(y_hat[1], target[1].view(-1, 1).float()) * 3
    loss3 = loss_fn(y_hat[2], target[2].view(-1, 1).float()) * 2
    loss4 = loss_fn(y_hat[3], target[3].view(-1, 1).float())
    loss = loss1 + loss2 + loss3 + loss4
    return loss


def train_loop(model, model_fn, data_reader):
    train_reader, valid_reader, test_reader = data_reader
    optimizer, scheduler, loss_fn, metric_fn = model_fn['optimizer'], model_fn['scheduler'], model_fn['loss_fn'], \
                                               model_fn['metric_fn']
    train_iter = 0
    for epoch in range(hparams['epoch_num']):
        train_data = train_reader.read()
        for idx1, train_dataloader in enumerate(train_data):
            for idx2, (X, y) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    for _ in range(5):
                        torch.cuda.empty_cache()
                scheduler(optimizer=optimizer, num_update=train_iter)
                y_hats = model(X)
                loss = get_loss(y_hats, y, loss_fn)
                writer.add_scalar('Loss/Train', loss, train_iter)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if train_iter % hparams['val_step'] == 0:
                    val_dataloader = valid_reader.read()
                    losses = []
                    for loader_i in val_dataloader:
                        for (val_X, val_y) in loader_i:
                            y_hats = model(val_X)
                            val_loss = get_loss(y_hats, val_y, loss_fn)
                            losses.append(val_loss.cpu().detach().item())
                    losses = np.array(losses)
                    writer.add_scalar('Loss/Valid', losses.mean(), train_iter)
                    print("epoch_%d/%d, data_%d, iter_%d, train_loss=%.2f, valid_loss=%.2f, lr=%g"
                          % (epoch + 1, hparams['epoch_num'],
                             idx1 + 1, idx2 + 1,
                             loss, losses.mean(), optimizer.state_dict()['param_groups'][0]['lr']))
                if train_iter % hparams['verbose_metric_step'] == 0:
                    test_loop(model, valid_reader, metric_fn)
                train_iter += 1
            # scheduler.step()
        if not os.path.exists(hparams['model_dir']):
            os.mkdir(hparams['model_dir'])
        torch.save(model.state_dict(), os.path.join(hparams['model_dir'], 'model_%s.pth' % epoch))


def test_loop(model, reader, metric_fn):
    test_dataloader = reader.read()
    target_dfs, result_dfs = [], []
    for loader_i in test_dataloader:
        for X, y in loader_i:
            y_hats = model(X)
            base_info = [e.view(-1, 1) for e in X[0:2]]
            result_ = torch.cat(base_info + y_hats, dim=1).cpu().detach().numpy()
            result_ = pd.DataFrame(result_, columns=['userid', 'feedid'] + ACTIONS)
            target_ = torch.cat(base_info + [e.view(-1, 1) for e in y], dim=1).cpu().detach().numpy()
            target_ = pd.DataFrame(target_, columns=['userid', 'feedid'] + ACTIONS)
            target_dfs.append(target_)
            result_dfs.append(result_)

    result_df = pd.concat(result_dfs, axis=0)
    target_df = pd.concat(target_dfs, axis=0)
    metric_fn(result_df, target_df)


def main(_argv):
    for dir in [hparams['output_dir'], hparams['model_dir'], hparams['log_dir']]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # 定义数据读入
    data_reader = [DataReader(hparams['train_dataset'], batch_size=hparams['batch_size'], shuffle=True,
                              chunksize=hparams['chunk_size']),
                   DataReader(hparams['test_dataset'], batch_size=hparams['batch_size'], shuffle=False,
                              nrows=hparams['val_size']),
                   DataReader(hparams['test_dataset'], batch_size=hparams['batch_size'], shuffle=False,
                              chunksize=hparams['chunk_size'])]

    if hparams['mode'] in ['offline_train', 'online_train']:
        side_information = pickle_process('load', hparams['feed_info_pro'])
        pretrain_emb = pickle_process('load', hparams['pretrain_embeddings'])
        model = DLRM(pretrain_emb, side_information)
        model.to(device)
        optimizer = get_optimizer(hparams['optimizer'])(model.parameters(), lr=hparams['base_lr'])
        model_fn = {
            'loss_fn': nn.BCEWithLogitsLoss(reduction='mean').to(device),
            'optimizer': optimizer,
            'scheduler': CosineScheduler(max_update=hparams['max_warm_batch'],
                                         base_lr=hparams['base_lr'],  # 预热阶段和cos阶段的交界值
                                         final_lr=hparams['final_lr'],  # cos阶段的最终值
                                         warmup_steps=hparams['warmup_steps'],  # 分多少步加热
                                         warmup_begin_lr=hparams['warmup_begin_lr'],  # 预热阶段的初始值
                                         warmup_mode=hparams['warmup_mode']).adjust_learning_rate,
            # 'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5], gamma=0.1),
            'metric_fn': score
        }

        # 模型训练
        logger.info("train model...")
        train_loop(model, model_fn, data_reader)
    elif 'val' in hparams['mode']:
        logger.info("upload model dict...")
        model = DLRM()
        model.load_state_dict(torch.load(os.path.join(hparams['model_dir'], 'model_%s.pth' % hparams['model_version']),
                                         map_location='cpu'), strict=True)
        model.to(device)

        logger.info("validation model...")
        test_loop(model, data_reader[2], score)


if __name__ == '__main__':
    app.run(main)
