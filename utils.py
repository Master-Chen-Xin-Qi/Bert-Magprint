# -*- coding: utf-8 -*-
"""
# @Time    : 2021/11/15 14:18
# @Author  : Xinqi Chen
# @Email   : chenxq66@sjtu.edu.cn
# @File    : utils.py
"""

# 加入了Magprint数据的载入


import argparse

import matplotlib.pyplot as plt
from scipy.special import factorial
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config, USERS, SIGMA\
    , WINDOW_LEN, MODEL_FOLDER


""" Utils Functions """

import random
import os
import numpy as np
import torch
import sys
import joblib


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


def prepare_pretrain_dataset(data, labels, training_rate, seed=None):
    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
                                                                                                  , training_rate=training_rate, vali_rate=0.1
                                                                                                  , change_shape=False)
    return data_train, label_train, data_vali, label_vali


def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False):

    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1
                                , change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(labels)
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                               , label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)


class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)


class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)


class MagprintDataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        return torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)

    def __len__(self):
        return len(self.data)


class MagprintDataset4Reconstruction(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, mask_data):
        super().__init__()
        self.data = data
        self.mask_data = mask_data
        self.seq_len = data.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.mask_data[index]), torch.arange(self.seq_len).long(), torch.from_numpy(self.data[index])

    def __len__(self):
        return len(self.data)


def handle_argv(target, config_train, prefix):
    parser = argparse.ArgumentParser(description='PyTorch Magprint-BERT Model')
    parser.add_argument('-model_version', type=str, help='Model config', default='v1')
    parser.add_argument('-dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'magprint'], default='magprint')
    parser.add_argument('-dataset_version',  type=str, help='Dataset version', choices=['10_240', '10_120'], default='10_240')  # 代表下采样率和时间窗长度
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')  # pretrain时default为None，这里改成了model.pt方便debug
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1,
                        help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='240_model',
                        help='The saved model name')
    try:
        args = parser.parse_args()
        model_cfg = load_model_config(target, prefix, args.model_version)
        if model_cfg is None:
            print("Unable to find corresponding model config!")
            sys.exit()
        args.model_cfg = model_cfg
        dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
        if dataset_cfg is None:
            print("Unable to find corresponding dataset config!")
            sys.exit()
        args.dataset_cfg = dataset_cfg
        args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, target=target)
        return args
    except:
        parser.print_help()
        sys.exit(0)


def handle_argv_simple():
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib','merge'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_240', '10_120'])
    args = parser.parse_args()
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    return args


def load_raw_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def load_pretrain_data_config(args):
    # pretrain.json和train.json中的barch size均改成了512
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg


def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


########################
# Magprint部分
########################
# 将名字转为label
def label_process(different_name):
    label_dict = dict()
    for i in range(len(different_name)):
        label_dict[different_name[i]] = i
    return label_dict


# Magprint专属数据集的载入，将不同user的数据分隔开
def divide_files_by_user_name(folder_name, different_name):
    dict_file = dict(zip(different_name, [] * len(different_name)))
    for user in different_name:
        dict_file[user] = []
        for (root, _, files) in os.walk(folder_name):
            for filename in files:
                if user in filename:
                    file = os.path.join(root, filename)
                    dict_file[user].append(file)
    return dict_file


# 读取每一个文件中的数据，可以进行下采样等操作
def read_single_file_data(single_file_name):
    # 每十个取一次平均，并且使数据长度为WINDOW_LEN整数倍，WINDOW_LEN*10个数据点作为一个时间窗
    mag_data = []
    fid = open(single_file_name, 'r')
    for line in fid:
        if '2018' in line:
            continue
        tmp = 0
        line = line.strip('\n')
        data = line.split(',')
        for i in range(len(data) - 1):
            tmp += int(data[i])
        mag_data.append(tmp//10)
    n = len(mag_data)//WINDOW_LEN
    total_num = n*WINDOW_LEN
    mag_data = np.array(mag_data[:total_num]).reshape(-1, WINDOW_LEN, 1)  # 每个WINDOW_LEN一组
    return mag_data, total_num


# 处理每一个文件数据，并最终保存
def process_file_data(dict_file, different_name, save_path):
    label_dict = label_process(different_name)
    all_data = np.zeros((1, WINDOW_LEN, 1))  # 储存所有用户的数据
    all_label = np.zeros((1, WINDOW_LEN, 1))
    for user in different_name:
        mag_data_total = np.zeros((1, WINDOW_LEN, 1))
        label = label_dict[user]
        label_total = np.zeros((1, WINDOW_LEN, 1))
        for single_file in dict_file[user]:
            mag_data, total_num = read_single_file_data(single_file)
            mag_data_total = np.vstack((mag_data_total, mag_data))
            reshaped_label = np.array([label]*total_num).reshape(-1, WINDOW_LEN, 1)
            label_total = np.vstack((label_total, reshaped_label))
        print('User:%s, Data Length: %d' %(user, len(mag_data_total)-1))
        mag_data_total = mag_data_total[1:]
        label_total = label_total[1:]
        all_data = np.vstack((all_data, mag_data_total))
        all_label = np.vstack((all_label, label_total))
        mag_data_total_pre = preprocess(mag_data_total)
        np.save(save_path+'/users_numpy/'+user+'_'+str(WINDOW_LEN), mag_data_total_pre)
        np.save(save_path + '/labels_numpy/' + user+'_'+str(WINDOW_LEN), label_total)
    all_data = all_data[1:]
    all_label = all_label[1:]
    np.save(save_path+'/users_numpy/'+'data'+'_10_'+str(WINDOW_LEN), all_data)
    np.save(save_path + '/labels_numpy/' + 'label'+'_10_'+str(WINDOW_LEN), all_label)
    return all_data, all_label


# FFT
def fft_transform(data):
    transformed = np.fft.fft(data)
    # transformed = np.abs(transformed)
    return transformed


# IFFT
def ifft_transform(data):
    transformed = np.fft.ifft(data)
    transformed = np.abs(transformed)
    return transformed


# 高斯滤波
def gaussian_filter(data, sigma):
    import scipy.ndimage
    gaussian_data = scipy.ndimage.filters.gaussian_filter1d(data, sigma)
    return gaussian_data


# 预处理，先FFT，再低通滤波滤除噪声，并IFFT回时域
def preprocess(data):
    fft_data = fft_transform(data)
    gaussian_data = gaussian_filter(fft_data, SIGMA)
    time_data = ifft_transform(gaussian_data)
    return time_data


# 归一化函数，这里对每一个时间窗做归一化，而不是对整体做归一化
def window_min_max(data):
    from sklearn import preprocessing
    data = data.reshape(-1, WINDOW_LEN)
    data = data.T  # 将数据转置，因为minmaxscaler是按列做标准化
    XX = preprocessing.MinMaxScaler().fit(data)
    min_max_data = XX.transform(data).T
    min_max_data = min_max_data.reshape(-1, WINDOW_LEN)
    joblib.dump(XX, MODEL_FOLDER + "/Window_Min_Max.m")
    return min_max_data.reshape(-1, WINDOW_LEN, 1)


# 对整体数据做归一化
def total_min_max(data):
    from sklearn import preprocessing
    data = data.reshape(-1, 1)
    XX = preprocessing.MinMaxScaler().fit(data)
    min_max_data = XX.transform(data)
    min_max_data = min_max_data.reshape(-1, WINDOW_LEN)
    joblib.dump(XX, MODEL_FOLDER + "/Total_Min_Max.m")
    return min_max_data.reshape(-1, WINDOW_LEN, 1)


# 整个流程汇总
def prepare_data(args, dict_file, different_name, save_path):
    d_f = divide_files_by_user_name(dict_file, different_name)
    all_data, all_label = process_file_data(d_f, different_name, save_path)
    # all_data = window_min_max(all_data)  # 对数据整体进行归一化操作
    np.save('./dataset/users_numpy/data_'+args.dataset_version+'.npy', all_data)
    print('ALL data have been saved!')


# if __name__ == '__main__':
#     all_data = np.load('./dataset/users_numpy/all_data.npy')
#     test = window_min_max(all_data)
#     np.save()