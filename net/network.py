import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer


class Network(Layer):
    def __init__(self,
                 sequence_vocabulary, bracket_vocabulary,
                 dmodel=128,
                 layers=8,
                 dropout=0.15,
                 stop_gradient=False):
        super(Network, self).__init__()
        self.sequence_vocabulary = sequence_vocabulary
        self.bracket_vocabulary = bracket_vocabulary
        self.dropout_rate = dropout
        self.model_size = dmodel
        self.layers = layers

        # Elmo模型相关配置
        self.stop_gradient = stop_gradient
        self.use_elmo = True
        self.use_bigru = False
        self.cell_clip = 3.0
        self.proj_clip = 3.0
        self.init_bound = 0.1
        self.bigru_num = 2

    def forward(self, seq, dot):
        """
        Forword 前馈神经网络
        :param seq:
        :param dot:
        :return:
        """
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True)

        if self.use_elmo:
            emb_seq_elmo = self.elmo_encoder(emb_seq)
            emb_dot_elmo = self.elmo_encoder(emb_dot)

            input_seq_feature = fluid.layers.concat(input=[emb_seq_elmo, emb_seq], axis=1)
            input_dot_feature = fluid.layers.concat(input=[emb_dot_elmo, emb_dot], axis=1)

            if self.use_bigru:
                print("Use bigru...")
                for i in range(self.bigru_num):
                    bigru_seq_output = self._bigru_layer(input_seq_feature)
                    input_seq_feature = bigru_seq_output
                    bigru_dot_output = self._bigru_layer(input_dot_feature)
                    input_dot_feature = bigru_dot_output

            emb_seq = fluid.layers.fc(input=input_seq_feature, size=self.model_size)
            emb_dot = fluid.layers.fc(input=input_dot_feature, size=self.model_size)

        emb = paddle.fluid.layers.concat(input=[emb_seq, emb_dot], axis=1)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        for _ in range(self.layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size * 4)
            fwd, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 4, use_peepholes=True,
                                                         is_reverse=False)
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 4, use_peepholes=True,
                                                          is_reverse=True)
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)
            emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        ff_out = paddle.fluid.layers.fc(emb, size=2, act="relu")
        soft_out = paddle.fluid.layers.softmax(ff_out, axis=1)
        return soft_out[:, 0]

    def elmo_encoder(self, x_emb):
        """
        Elmo神经网络
        :param emb:
        :return:
        """

        # 数据翻转
        x_emb_r = fluid.layers.sequence_reverse(x_emb)
        fw_hiddens, fw_hiddens_ori = self.encoder_wapper(x_emb, self.model_size)
        bw_hiddens, bw_hiddens_ori = self.encoder_wapper(x_emb_r, self.model_size)

        num_layers = len(fw_hiddens_ori)
        token_embeddings = fluid.layers.concat(input=[x_emb, x_emb], axis=1)
        # token_embeddings.stop_gradient = self.stop_gradient  # 是否停止梯度更新
        concate_embeddings = [token_embeddings]

        for index in range(num_layers):
            embedding = fluid.layers.concat(
                input=[fw_hiddens_ori[index], bw_hiddens_ori[index]], axis=1)
            embedding = self.dropout(embedding)
            # embedding.stop_gradient = self.stop_gradient  # 是否停止梯度更新
            concate_embeddings.append(embedding)
        weighted_emb = self.weight_layers(concate_embeddings)
        return weighted_emb

    def encoder_wapper(self, x_emb, emb_size, init_hidden=None, init_cell=None):
        """
        Input:
            x_emb
        Output:
            x_emb --> LSTM --> rnn_out
            rnn_out_ori = rnn_out

            if i > 0:
                rnn_out = rnn_out + rnn_input

            rnn_outs append rnn_out
            rnn_outs_ori append rnn_out_ori
        :param x_emb:
        :param emb_size:
        :param init_hidden:
        :param init_cell:
        :return:
        """
        rnn_input = x_emb
        rnn_outs = []
        rnn_outs_ori = []
        cells = []
        projs = []
        num_layers = 2  # Elmo每层LSTM数量

        for i in range(num_layers):
            if init_hidden and init_cell:
                h0 = fluid.layers.squeeze(
                    fluid.layers.slice(init_hidden, axes=[0], starts=[i], ends=[i + 1]),
                    axes=[0])
                c0 = fluid.layers.squeeze(
                    fluid.layers.slice(init_cell, axes=[0], starts=[i], ends=[i + 1]),
                    axes=[0])
            else:
                h0 = c0 = None
            rnn_out, cell, input_proj = self.lstmp_encoder(rnn_input, self.model_size, h0, c0, emb_size)
            rnn_out_ori = rnn_out
            if i > 0:
                rnn_out = rnn_out + rnn_input
            # rnn_out.stop_gradient = self.stop_gradient  # 是否停止梯度更新
            rnn_outs.append(rnn_out)
            rnn_outs_ori.append(rnn_out_ori)
        return rnn_outs, rnn_outs_ori

    def lstmp_encoder(self, input_seq, gate_size, h_0, c_0, proj_size):
        """
        LSTM
        :param input_seq:
        :param gate_size:
        :param h_0:
        :param c_0:
        :param proj_size:
        :return:
        """
        input_proj = fluid.layers.fc(input=input_seq, size=gate_size * 4)

        hidden, cell = fluid.layers.dynamic_lstmp(
            input=input_proj,
            size=gate_size * 4,
            proj_size=proj_size,
            h_0=h_0,
            c_0=c_0,
            use_peepholes=False,
            proj_clip=self.proj_clip,
            cell_clip=self.cell_clip,
            proj_activation='identity')
        return hidden, cell, input_proj

    def weight_layers(self, lm_embeddings):
        n_lm_layers = len(lm_embeddings)
        W = fluid.layers.create_parameter([n_lm_layers, ], dtype="float32")
        normed_weights = fluid.layers.softmax(W + 1.0 / n_lm_layers)
        splited_normed_weights = fluid.layers.split(normed_weights, n_lm_layers, dim=0)

        pieces = []
        for w, t in zip(splited_normed_weights, lm_embeddings):
            pieces.append(t * w)
        sum_pieces = fluid.layers.sums(pieces)

        gamma = fluid.layers.create_parameter([1], dtype="float32")
        weighted_lm_layers = sum_pieces * gamma
        return weighted_lm_layers

    def dropout(self, input):
        return fluid.layers.dropout(input, dropout_prob=self.dropout_rate, is_test=False)

    def _bigru_layer(self, input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=self.model_size * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=self.model_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=self.model_size * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=self.model_size,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge
