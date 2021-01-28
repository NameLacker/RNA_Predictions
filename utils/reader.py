import paddle.fluid as fluid
import paddle
import numpy as np
import re

from config import RNA_Config

collocations = RNA_Config()


def train_reader():
    """
    训练数据读取
    :return:
    """
    global score
    train_data_file = collocations.train_dataset
    with open(train_data_file, 'r') as f:
        train_data = f.readlines()
    lock = 0
    rnas = []
    labels = []
    scores = []
    for line in train_data:
        if line == '\n':
            scores.append(score)
        lock += 1
        if line[:3] == '>id':
            lock = 0
            score = []
        if lock == 1:
            rnas.append(line.replace("\n", ""))
        if lock == 2:
            labels.append(line.replace("\n", ""))
        if lock > 2:
            score.append(line.replace("\n", ""))

    train_data = zip(rnas, labels, scores)
    return train_data


def val_reader():
    """
    验证数据读取
    :return:
    """
    pass


def test_reader():
    """
    测试数据读取
    :return:
    """
    pass
