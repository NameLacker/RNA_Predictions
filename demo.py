"""
测试
"""
import json
import paddle
import paddle.fluid as fluid

from utils.reader import load_train_data
from utils.process import process_vocabulary
from net.network import Network
from config import RNA_Config
from utils.reader import read_data

collocations = RNA_Config()


def get_rna():
    """
    提取rna序列并写入rna.txt文件
    :return:
    """
    trains, devs = load_train_data()
    with open('data/rna.txt', 'w') as f:
        for train in trains:
            f.write(str(train["sequence"]) + '\n')


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
        print("No.{}\tscore: {:.3}\t\tavg: {}\tstd: {}".format(n+1, score, avg, std))
    while True:
        avg, std = input().split()
        avg = float(avg)
        std = float(std)
        print("avg: {}\nstd: {}".format(avg, std))
        score = 0.9609 / avg + 0.0102 / std
        print("score: {}".format(score))


def mix_rna():

    pass


if __name__ == '__main__':
    get_rna()
    #　parallel()
    # read_size()
    # opt_score()
    # mix_rna()
    pass
