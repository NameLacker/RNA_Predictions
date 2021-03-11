"""
测试
"""
import json
import paddle
import paddle.fluid as fluid
import re
import random

from utils.reader import load_train_data
from utils.process import process_vocabulary
from net.network import Network
from config import RNA_Config
from utils.reader import read_data
import pahelix.toolkit.linear_rna as linear_rna

collocations = RNA_Config()


class RNA_Stack:
    def __init__(self):
        self.data = []  # 储存下标

    def push(self, index):
        """
        入栈
        :return:
        """
        self.data.append(index)

    def pop(self):
        """
        出栈
        :return:
        """
        size = len(self.data) - 1
        return self.data.pop(size)


def get_rna():
    """
    提取rna序列并写入rna.txt文件
    :return:
    """
    trains, devs = load_train_data()
    '''
    with open('data/rna.txt', 'w') as f:
        for train in trains:
            f.write(str(train["sequence"]) + '\n')
            '''
    return trains


def parallel():
    """
    生成仅含RNA序列的txt文件
    :return:
    """
    trains, devs = load_train_data()

    file = "data/rna.txt"
    with open(file, 'r') as f:
        rna = f.readlines()
    rna = rna[1::2]

    print(len(rna))
    print(len(trains))
    index = 4750
    with open("./data/other_train.txt", 'w') as f:
        for train, new in zip(trains, rna):

            rna_1 = train["structure"]
            rna_2 = new.split()[0]

            if rna_1 != rna_2:
                index += 1
                f.write(">id_" + str(index) + "\n")
                f.write(str(train["sequence"]) + "\n")
                f.write(str(rna_2) + "\n")
                for id, score in enumerate(train["p_unpaired"]):
                    id += 1
                    f.write(str(id) + " " + str(score) + "\n")
                f.write("\n")


def read_size():
    """
    读取碱基序列中的长度最大值
    :return:
    """
    dataset = read_data("./data/test_nolabel.txt", True)
    max = 0
    for data in dataset:
        size = len(data["sequence"])
        if size > max:
            max = size
    print(max)


def read_score():
    """
    读取score
    :return:
    """
    with open("scores", 'r') as f:
        scores = f.readlines()
    for i, score in enumerate(scores):
        scores[i] = [eval(d) for d in score.split()]
    return scores


def opt_score():
    """
    计算score计算过程
    :return:
    """
    scores = read_score()
    for n, score in enumerate(scores):
        sc, avg, std = score
        score = 0.9159 / avg - 0.00466 / std + 0.4879
        print("No.{}\tscore: {:.3}\t\tavg: {}\tstd: {}".format(n + 1, score, avg, std))
    while True:
        avg, std = input().split()
        avg = float(avg)
        std = float(std)
        print("avg: {}\nstd: {}".format(avg, std))
        score = 0.9609 / avg + 0.0102 / std
        print("score: {}".format(score))


def reverse_rna():
    """
    数据翻转
    :return:
    """
    trains = get_rna()
    rev_data = []
    for train in trains:
        temp = {}
        seq = train['sequence']
        dot = train['structure']
        prob = train['p_unpaired']

        seq_rev = seq[::-1]
        dot_rev = dot[::-1]
        dot_rev = dot_rev.replace('(', 't')
        dot_rev = dot_rev.replace(')', '(')
        dot_rev = dot_rev.replace('t', ')')
        prob_rev = prob[::-1]
        temp['sequence'] = seq_rev
        temp['structure'] = dot_rev
        temp['p_unpaired'] = prob_rev
        rev_data.append(temp)

    max_size = len(rev_data) - 1
    with open("./data/rev_train.txt", 'w') as f:
        for id, rev in enumerate(rev_data):
            f.write('>id_' + str(id + 1) + '\n')
            f.write(rev['sequence'] + '\n')
            f.write(rev['structure'] + '\n')
            for n, up in enumerate(rev['p_unpaired']):
                f.write(str(n + 1) + ' ' + str(up) + '\n')
            if id < max_size:
                f.write('\n')


def exchange_rna():
    """
    同源序列互换
    :return:
    """
    trains = get_rna()
    ex_data = []
    rna_stack = RNA_Stack()
    for train in trains:
        temp = {}
        seq = train['sequence']
        dot = train['structure']
        prob = train['p_unpaired']

        seq_list = list(seq)
        index = 0
        suffix = 0
        # 同源互换
        for s, d, p in zip(seq_list, dot, prob):
            lock = 0
            if d == '(':
                rna_stack.push(index)
            if d == ')':
                suffix = rna_stack.pop()
                lock = 1
            if random.random() >= 0.5 and lock == 1:
                tmp_s = seq_list[index]
                seq_list[index] = seq_list[suffix]
                seq_list[suffix] = tmp_s

                tmp_p = prob[index]
                prob[index] = prob[suffix]
                prob[suffix] = tmp_p
            index += 1
        seq = "".join(seq_list)
        temp['sequence'] = seq
        temp['structure'] = dot
        temp['p_unpaired'] = prob
        ex_data.append(temp)

    max_size = len(ex_data) - 1
    with open("./data/ex_train.txt", 'w') as f:
        for id, ex in enumerate(ex_data):
            f.write('>id_' + str(id + 1) + '\n')
            f.write(ex['sequence'] + '\n')
            f.write(ex['structure'] + '\n')
            for n, up in enumerate(ex['p_unpaired']):
                f.write(str(n + 1) + ' ' + str(up) + '\n')
            if id < max_size:
                f.write('\n')


if __name__ == '__main__':
    # get_rna()
    # parallel()
    # read_size()
    # opt_score()
    # reverse_rna()
    exchange_rna()
    pass
