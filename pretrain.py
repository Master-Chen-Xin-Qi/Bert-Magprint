#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig, USERS, WINDOW_LEN
from models import MagprintModel4Pretrain
from utils import set_seeds, get_device \
    , MagprintDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask, prepare_data


def main(args, training_rate):
    # prepare_data(args, './dataset/users', USERS, './dataset')  # 这里是载入并预处理magprint数据，运行一次即可
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    # 原始数据太大，为(557075,120,1)，先截取1/100做测试
    data = data[:-1:50, :, :]
    labels = labels[:-1:50, :, :]

    # 磁信号幅值太大，会导致梯度爆炸问题，这里需要对输入进行约束
    data = data/100
    # data = data.reshape(-1, 240)
    # min_data = np.min(data)
    # max_data = np.max(data)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    # pipeline = [Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)
    data_train_np, data_test_np = data_train.reshape(-1, WINDOW_LEN), data_test.reshape(-1, WINDOW_LEN)
    # 制作Dataset，每一个数据都要进行pipeline中的处理：1.标准化；2.mask
    data_set_train = MagprintDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = MagprintDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = MagprintModel4Pretrain(model_cfg)

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)

    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch  # 三个分别是被mask后的序列，mask的位置，原始序列
        mask_seqs_numpy, masked_pos_numpy, seqs_numpy = mask_seqs.numpy().reshape(-1, WINDOW_LEN), masked_pos.numpy(), seqs.numpy().reshape(-1, int(WINDOW_LEN*0.15))
        seq_recon = model(mask_seqs, masked_pos)  # 模型的输入是被mask的数据和mask的位置
        seq_recon_numpy = seq_recon.detach().numpy().reshape(-1, int(WINDOW_LEN*0.15))
        # 实时观察数据
        np.save('BERT实时预测的结果', seq_recon_numpy)
        np.save('真实数据', seqs_numpy)
        loss_lm = criterion(seq_recon, seqs)  # for masked LM
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
                      , model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)

