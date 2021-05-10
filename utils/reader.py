import os
import numpy as np
import random
from config import RNA_Config
from utils.utils import reverse, cross_over

collocations = RNA_Config()


def read_data(filename, test=False):
    """
    返回data字典对象
        {
            "id": >id_13,
            "sequence": UAGGGAAUAUUGGGCA...,
            "structure": ...((....(((((..........,
            "p_unpaired": [[1, 0.2461], [2, 0.4946], ......],
        }
    :param filename:
    :param test:
    :return:
    """
    data, x = [], []
    for line in open(filename, "r"):
        line = line.strip()
        if not line:
            ID, seq, dot = x[:3]
            if test:
                x = {"id": ID,
                     "sequence": seq,
                     "structure": dot,
                     }
                data.append(x)
                x = []
                continue
            punp = x[3:]
            punp = [punp_line.split() for punp_line in punp]
            punp = [(float(p)) for i, p in punp]
            x = {"id": ID,
                 "sequence": seq,
                 "structure": dot,
                 "p_unpaired": punp,
                 }

            data.append(x)
            x = []
        else:
            x.append(line)
    return data


def reader_creator(data,
                   sequence_vocabulary, bracket_vocabulary,
                   test=False):
    def reader():
        for i, x in enumerate(data):
            seq = x["sequence"]
            dot = x["structure"]
            seq = [sequence_vocabulary.index(x) for x in list(seq)]
            dot = [bracket_vocabulary.index(x) for x in list(dot)]

            if not test:
                LP_v_unpaired_prob = x["p_unpaired"]
                prob = [x for x in LP_v_unpaired_prob]

                if random.random() > 0.66:  # 随机跳过
                    continue
                if random.random() > 0.5:  # 随机数据翻转
                    seq, dot, prob = reverse(seq, dot, prob)
                if random.random() > 0.5:  # 随机同源序列互换
                    seq, dot, prob = cross_over(seq, dot, prob)

                sequence = np.array(seq)
                structure = np.array(dot)
                LP_v_unpaired_prob = np.array(prob)
                yield sequence, structure, LP_v_unpaired_prob
            else:
                sequence = np.array(seq)
                structure = np.array(dot)
                yield sequence, structure

    return reader


def load_train_data():
    """
    训练数据读取器
    :return:
    """
    assert os.path.exists(collocations.train_dataset)
    assert os.path.exists(collocations.train_dataset_other)
    assert os.path.exists(collocations.train_dataset_reverse)
    assert os.path.exists(collocations.train_dataset_exchange)
    assert os.path.exists(collocations.dev_dataset)

    train = None
    if collocations.add == 0:
        print("Load original dataset...")
        train = read_data(collocations.train_dataset)
    elif collocations.add == 1:
        print("Load augmentation dataset...")
        train = read_data(collocations.train_dataset_other)
    if collocations.add == 2:
        print("Load original and augmentation dataset...")
        train = read_data(collocations.train_dataset)
        train1 = read_data(collocations.train_dataset_other)
        train.extend(train1)
    if collocations.add == 3:
        print("Load reverse dataset...")
        train = read_data(collocations.train_dataset_exchange)
    if collocations.add == 4:
        print("Load all dataset...")
        train = read_data(collocations.train_dataset)
        train1 = read_data(collocations.train_dataset_exchange)
        train.extend(train1)
    dev = read_data(collocations.dev_dataset)
    assert train is not None
    assert dev is not None
    return train, dev


def load_test_A():
    """
    读取A榜测试数据
    :return:
    """
    assert os.path.exists(collocations.test_A)
    test = read_data(collocations.test_A, test=True)
    return test


def load_test_B():
    """
    读取B榜测试数据
    :return:
    """
    assert os.path.exists(collocations.test_B)
    test = read_data(collocations.test_B, test=True)
    return test
