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
    train_data_file = collocations.train_dataset
    with open(train_data_file, 'r') as f:
        train_data = f.readlines()
    lock = 0
    rnas = []
    labels = []
    scores = []
    for line in train_data:
        try:
            if line == '\n':
                scores.append(score)
            lock += 1
            if line[:3] == '>id':
                lock = 0
                score = []
            if lock == 1:
                rnas.append(line.replace("\n", ""))  # RNA序列
            if lock == 2:
                labels.append(line.replace("\n", ""))  # 二级配对关系
            if lock > 2:
                score.append(eval(line.replace("\n", "").split()[1]))  # 各个位置未配对概率
        except Exception as e:
            pass
    train_data = zip(rnas, labels, scores)

    def reader():
        for rna, label, score in train_data:
            rna = np.array([collocations.input[base] for base in rna])  # 字符映射 A->0, U->1, C->2, G->3
            label = np.array([collocations.label[structure] for structure in label])  # 字符映射 '('->0, ')'->1, '.'->2
            score = np.array(score)
            yield rna, label, score
    return reader()


def val_reader():
    """
    验证数据读取
    :return:
    """
    train_data_file = collocations.dev_dataset
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

    def reader():
        for rna, label, score in train_data:
            yield rna, label

    return reader()


def test_reader():
    """
    测试数据读取
    :return:
    """
    pass
