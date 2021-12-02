# -*- coding: utf-8 -*-
"""
# @Time    : 2021/11/29 13:58
# @Author  : Xinqi Chen
# @Email   : chenxq66@sjtu.edu.cn
# @File    : 检查训练.py
"""
import numpy as np
real = np.load('真实数据.npy')
predict = np.load('BERT实时预测的结果.npy')
print('finish')