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
    train_datas = zip(rnas, labels, scores)

    def reader():
        for rna, label, score in train_datas:
            size = len(rna)
            null = [0 for i in range(500 - size)]
            rna = np.concatenate((np.array([collocations.input[base] for base in rna]), null),
                                 axis=0)  # 字符映射 A->1, U->2, C->3, G->4
            label = np.concatenate((np.array([collocations.label[structure] for structure in label]), null),
                                   axis=0)  # 字符映射 '('->1, ')'->2, '.'->3
            rna = np.concatenate((rna, label), axis=0)
            score = np.concatenate((np.array(score), null), axis=0)
            yield rna, score
    return reader


def val_reader():
    """
    验证数据读取
    :return:
    """
    val_data_file = collocations.dev_dataset
    with open(val_data_file, 'r') as f:
        val_data = f.readlines()
    lock = 0
    rnas = []
    labels = []
    scores = []
    for line in val_data:
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
    val_datas = zip(rnas, labels, scores)

    def reader():
        for rna, label, score in val_datas:
            size = len(rna)
            null = [0 for i in range(500 - size)]
            rna = np.concatenate((np.array([collocations.input[base] for base in rna]), null),
                                 axis=0)  # 字符映射 A->1, U->2, C->3, G->4
            label = np.concatenate((np.array([collocations.label[structure] for structure in label]), null),
                                   axis=0)  # 字符映射 '('->1, ')'->2, '.'->3
            rna = np.concatenate((rna, label), axis=0)
            score = np.concatenate((np.array(score), null), axis=0)
            yield rna, score

    return reader


def test_reader():
    """
    测试数据读取
    :return:
    """
    pass
