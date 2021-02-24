import paddle.fluid as fluid

from config import RNA_Config

collocations = RNA_Config()


def optimizer():
    # 动态学习率
    boundaries = [step * collocations.each_step for step in collocations.boundaries]
    values = [value * collocations.learn_rate for value in collocations.values]
    learn_rate = fluid.layers.piecewise_decay(boundaries, values)
    # 优化方法
    optimize = adam(learn_rate)  # TODO: 共有8个优化方法可供选择
    return optimize, learn_rate


def sgd(learning_rate):
    """
    该接口实现随机梯度下降算法的优化器
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.SGD(learning_rate=learning_rate,
                               regularization=fluid.regularizer.L2Decay(0.00005))


def momentum(learning_rate):
    """
    该接口实现含有速度状态的Simple Momentum 优化器
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.Momentum(learning_rate=learning_rate,
                                    momentum=0.9,
                                    regularization=fluid.regularizer.L2Decay(0.00005))


def adagrad(learning_rate):
    """
    Adaptive Gradient 优化器(自适应梯度优化器，简称Adagrad)
    可以针对不同参数样本数不平均的问题，自适应地为各个参数分配不同的学习率。
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.Adagrad(learning_rate=learning_rate,
                                   epsilon=collocations.epsilon,
                                   regularization=fluid.regularizer.L2Decay(0.00005))


def rmsprop(learning_rate):
    """
    该接口实现均方根传播（RMSProp）法，是一种未发表的,自适应学习率的方法。
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.RMSProp(learning_rate=learning_rate,
                                   epsilon=collocations.epsilon,
                                   momentum=0.9,
                                   centered=True,
                                   regularization=fluid.regularizer.L2Decay(0.00005))


def adam(learning_rate):
    """
    Adam优化器出自 Adam论文 的第二节，能够利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.Adam(learning_rate=learning_rate,
                                beta1=collocations.beta1,
                                beta2=collocations.beta2,
                                epsilon=collocations.epsilon)


def adamax(learning_rate):
    """
    Adamax优化器是参考 Adam论文 第7节Adamax优化相关内容所实现的。
    Adamax算法是基于无穷大范数的 Adam 算法的一个变种，使学习率更新的算法更加稳定和简单。
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.Adamax(learning_rate=learning_rate,
                                  beta1=collocations.beta1,
                                  beta2=collocations.beta2,
                                  epsilon=collocations.epsilon,
                                  regularization=fluid.regularizer.L2Decay(0.00005))


def ftrl(learning_rate):
    """
    该接口实现FTRL (Follow The Regularized Leader) Optimizer.
    :param learning_rate:
    :return:
    """
    return fluid.optimizer.Ftrl(learning_rate=learning_rate,
                                regularization=fluid.regularizer.L2Decay(0.00005))


def modelaverage(average_window_rate=0.15, min_average_window=10000, max_average_window=12500):
    """
    ModelAverage优化器，在训练过程中累积特定连续的历史Parameters，
    累积的历史范围可以用传入的average_window参数来控制，在预测时使用平均后的Parameters，通常可以提高预测的精度。
    :param average_window_rate:
    :param min_average_window:
    :param max_average_window:
    :return:
    """
    return fluid.optimizer.ModelAverage(average_window_rate=average_window_rate,
                                        min_average_window=min_average_window,
                                        max_average_window=max_average_window,
                                        regularization=fluid.regularizer.L2Decay(0.00005))
