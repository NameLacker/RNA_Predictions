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
    label = fluid.layers.data(name="score", shape=[500], dtype="float64")
    avg_cost = cost_function(prediction, label)
    # accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost]  # 返回平均cost和acc


def cost_function(prediction, label):
    """
    自定义交叉熵损失函数
    :param prediction:
    :param label:
    :return:
    """
    sub = fluid.layers.elementwise_sub(prediction, label)
    cost = fluid.layers.square(sub)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    """
    损失函数
    :return:
    """
    boundaries = [step * 200 for step in collocations.boundaries]
    values = [value * collocations.learn_rate for value in collocations.values]
    learn_rate = fluid.layers.piecewise_decay(boundaries, values)
    return fluid.optimizer.RMSPropOptimizer(learning_rate=learn_rate), learn_rate
