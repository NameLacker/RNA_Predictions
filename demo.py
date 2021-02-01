"""
测试
"""
import paddle
import paddle.fluid as fluid

from utils.reader import *

trains, devs = load_train_data()
for train, dev in zip(trains, devs):
    print(train)
    print(dev)
    break
