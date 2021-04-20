import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer

from net.bilm import *


class Network(Layer):
    def __init__(self,
                 sequence_vocabulary, bracket_vocabulary,
                 dmodel=128,
                 layers=8,
                 dropout=0.15):
        super(Network, self).__init__()
        self.sequence_vocabulary = sequence_vocabulary
        self.bracket_vocabulary = bracket_vocabulary
        self.dropout_rate = dropout
        self.model_size = dmodel
        self.layers = layers

    def initialization(self):
        """
        参数初始化
        :return: 初始化的参数
        """
        end_train = True
        if end_gradient:
            end_train = False
        return fluid.ParamAttr(trainable=end_train)

    def forward(self, seq, dot):
        """
        Forword 前馈神经网络
        :param seq:
        :param dot:
        :return:
        """
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True,
                                         param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0))
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True,
                                         param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0))

        if use_elmo:
            emb_seq_elmo = elmo_encoder(emb_seq)
            emb_dot_elmo = elmo_encoder(emb_dot)

            input_seq_feature = fluid.layers.concat(input=[emb_seq_elmo, emb_seq], axis=1)
            input_dot_feature = fluid.layers.concat(input=[emb_dot_elmo, emb_dot], axis=1)

            if use_bigru:
                print("Use bigru...")
                for i in range(bigru_num):
                    bigru_seq_output = bigru_layer(input_seq_feature)
                    input_seq_feature = bigru_seq_output
                    bigru_dot_output = bigru_layer(input_dot_feature)
                    input_dot_feature = bigru_dot_output

            emb_seq = fluid.layers.fc(input=input_seq_feature, size=self.model_size)
            emb_dot = fluid.layers.fc(input=input_dot_feature, size=self.model_size)
            if stop_gradient:
                # 使前面的梯度停止更新
                emb_seq.stop_gradient = True
                emb_dot.stop_gradient = True

        emb = paddle.fluid.layers.concat(input=[emb_seq, emb_dot], axis=1)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu",
                                     param_attr=self.initialization(), bias_attr=self.initialization())
        for _ in range(self.layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size * 4,
                                         param_attr=self.initialization(), bias_attr=self.initialization())
            fwd, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 4, use_peepholes=True,
                                                         is_reverse=False, param_attr=self.initialization(),
                                                         bias_attr=self.initialization())
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 4, use_peepholes=True,
                                                          is_reverse=True, param_attr=self.initialization(),
                                                          bias_attr=self.initialization())
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)
            emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu",
                                         param_attr=self.initialization(), bias_attr=self.initialization())
        emb = dropout(emb, )
        ff_out = paddle.fluid.layers.fc(emb, size=2, act="relu",
                                        param_attr=self.initialization(), bias_attr=self.initialization())
        soft_out = paddle.fluid.layers.softmax(ff_out, axis=1)
        return soft_out[:, 0]
