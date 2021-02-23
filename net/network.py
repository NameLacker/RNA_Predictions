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

    def forward(self, seq, dot):
        """
        Forword 前馈神经网络
        :param seq:
        :param dot:
        :return:
        """
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True)

        if use_elmo:
            emb_seq_elmo = elmo_encoder(emb_seq)
            emb_dot_elmo = elmo_encoder(emb_dot)

            input_seq_feature = fluid.layers.concat(input=[emb_seq_elmo, emb_seq], axis=1)
            input_dot_feature = fluid.layers.concat(input=[emb_dot_elmo, emb_dot], axis=1)

            if use_bigru:
                print("Use bigru...")
                for i in range(self.bigru_num):
                    bigru_seq_output = bigru_layer(input_seq_feature)
                    input_seq_feature = bigru_seq_output
                    bigru_dot_output = bigru_layer(input_dot_feature)
                    input_dot_feature = bigru_dot_output

            emb_seq = fluid.layers.fc(input=input_seq_feature, size=self.model_size)
            emb_dot = fluid.layers.fc(input=input_dot_feature, size=self.model_size)

        emb = paddle.fluid.layers.concat(input=[emb_seq, emb_dot], axis=1)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu", name="1_fc")
        max_num = 3
        for _ in range(self.layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size * 4, name="{}_fc".format(1 + _*3 + 1))
            fwd, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 4, use_peepholes=True,
                                                         is_reverse=False, name="{}_0_lstm".format(1 + _*3 + 2))
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 4, use_peepholes=True,
                                                          is_reverse=True, name="{}_1_lstm".format(1 + _*3 + 2))
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)
            emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu", name="{}_fc".format(1 + _*3 + 3))
            max_num = 1 + _*3 + 3
        emb = dropout(emb)
        ff_out = paddle.fluid.layers.fc(emb, size=2, act="relu", name="{}_fc".format(max_num + 1))
        soft_out = paddle.fluid.layers.softmax(ff_out, axis=1, name="{}_softmax".format(max_num + 2))
        return soft_out[:, 0]
