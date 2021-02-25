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
    dataset = read_data("./data/test_nolabel.txt", True)
    max = 0
    for data in dataset:
        size = len(data["sequence"])
        if size > max:
            max = size
    print(max)


if __name__ == '__main__':
    # get_rna()
    # parallel()
    read_size()
    pass
