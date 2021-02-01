import os
import numpy as np


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


def load_train_data():
    assert os.path.exists("data/train.txt")
    assert os.path.exists("data/dev.txt")
    train = read_data("data/train.txt")
    dev = read_data("data/dev.txt")
    return train, dev


def load_test_data():
    assert os.path.exists("data/test_nolabel.txt")
    test = read_data("data/test_nolabel.txt", test=True)
    return test


def load_test_label_data():
    assert os.path.exists("data/test.txt")
    test = read_data("data/test.txt")
    return test


def reader_creator(args, data,
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
