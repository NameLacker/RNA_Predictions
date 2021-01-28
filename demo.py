"""
测试
"""
import paddle
import paddle.fluid as fluid

from utils.reader import *

traindata = train_reader()
for rna, label, score in traindata:
    print(rna)
    print(label)
    print(score)
    break
