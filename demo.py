"""
测试
"""
import paddle
import paddle.fluid as fluid

from utils.reader import load_train_data
from utils.process import process_vocabulary

trains, devs = load_train_data()
for train, dev in zip(trains, devs):
    # print(train)
    # print(dev)
    break

seq_vocab, bracket_vocab = process_vocabulary(trains)
print(seq_vocab)
print(bracket_vocab)
