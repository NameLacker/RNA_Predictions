import paddle.fluid as fluid
from net.network import *
from config import Config

collocations = Config()


def inference_program():
    """
    预测程序
    :param word_list:
    :return:
    """
    data = fluid.layers.data(name="rna", shape=[1], dtype="int64", lod_level=1)

    # net = convlution_net(data, dict_dim,
    #                      collocations.class_dim,
    #                      collocations.emb_dim, collocations.hid_dim)
    net = stacked_lstm_net(data)
    return net


def train_program(prediction):
    """
    训练程序
    :param prediction:
    :return:
    """
    label = fluid.layers.data(name="score", shape=[1], dtype="int64")
    length = fluid.layers.data(name="length", shape=[1], dtype="int64")
    # cost = fluid.layers.cross_entropy(input=prediction, label=label)
    cost = fluid.layers.dice_loss(prediction, label)
    avg_cost = fluid.layers.mean(cost)
    # accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost]  # 返回平均cost和acc


def optimizer_func():
    """
    损失函数
    :return:
    """
    boundaries = [step * 200 for step in collocations.boundaries]
    values = [value * collocations.learn_rate for value in collocations.values]
    learn_rate = fluid.layers.piecewise_decay(boundaries, values)
    return fluid.optimizer.Adagrad(learning_rate=learn_rate), learn_rate
