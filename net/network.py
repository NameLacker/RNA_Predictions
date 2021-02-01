from __future__ import print_function
import paddle.fluid as fluid

from config import RNA_Config

paramerts = RNA_Config()


def stacked_lstm_net(rna, label):
    """
    栈式双向LSTM结构
    :param rna:
    :param label:
    :return:
    """
    stacked_num = paramerts.stacked_num
    input_dim = paramerts.max_size
    emb_dim = paramerts.emb_dim
    hid_dim = paramerts.hid_dim
    class_dim = paramerts.class_dim
    assert stacked_num % 2 == 1
    # 计算词向量
    emb_rna = fluid.layers.embedding(
        input=rna, size=[input_dim, emb_dim], is_sparse=True)
    emb_label = fluid.layers.embedding(
        input=label, size=[input_dim, emb_dim], is_sparse=True
    )
    # 第一层栈
    # 全连接层
    fc1_rna = fluid.layers.fc(input=emb_rna, size=hid_dim)
    fc1_label = fluid.layers.fc(input=emb_label, size=hid_dim)
    fc1 = fluid.layers.concat([fc1_rna, fc1_label], axis=1)
    fc1 = fluid.layers.fc(input=fc1, size=hid_dim)
    # lstm层
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    # 其余所有栈结构
    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    # 池化层
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    # 全连接层，softmax预测
    prediction = fluid.layers.fc(input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction
