import os
import numpy as np
from config import RNA_Config

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
            sequence = np.array([sequence_vocabulary.index(x) for x in list(seq)])
            structure = np.array([bracket_vocabulary.index(x) for x in list(dot)])
            if not test:
                LP_v_unpaired_prob = x["p_unpaired"]
                LP_v_unpaired_prob = np.array([x for x in LP_v_unpaired_prob])
                yield sequence, structure, LP_v_unpaired_prob
            else:
                yield sequence, structure

    return reader


def load_train_data():
    """
    训练数据读取器
    :return:
    """
    assert os.path.exists(collocations.train_dataset)
    assert os.path.exists(collocations.train_dataset_other)
    assert os.path.exists(collocations.dev_dataset)

    train = read_data(collocations.train_dataset)
    train1 = read_data(collocations.train_dataset_other)
    train.extend(train1)
    dev = read_data(collocations.dev_dataset)
    return train, dev


def load_test_data():
    """
    不带标签的测试数据读取器
    :return:
    """
    assert os.path.exists(collocations.test_dataset)
    test = read_data(collocations.test_dataset, test=True)
    return test


def load_test_label_data():
    """
    由于比赛的公开数据不提供测试集的标签，故本模型无法运行预设的test_withlabel，
    需自己生成一个带标签的测试集~/data/test.txt
    :return:
    """
    assert os.path.exists(collocations.test)
    test = read_data(collocations.test)
    return test
