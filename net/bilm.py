import paddle.fluid as fluid

from config import RNA_Config

collocations = RNA_Config()

model_size = collocations.dmodel
stop_gradient = collocations.pre_training
end_gradient = collocations.end_training
use_elmo = collocations.use_elmo
use_bigru = collocations.use_bigru
dropout_rate = collocations.dropout
cell_clip = collocations.cell_clip
proj_clip = collocations.proj_clip
init_bound = collocations.init_bound
bigru_num = collocations.bigru_num


def elmo_encoder(x_emb):
    """
    Elmo神经网络
    :param emb:
    :return:
    """
    # 数据翻转
    x_emb_r = fluid.layers.sequence_reverse(x_emb)
    fw_hiddens, fw_hiddens_ori = encoder_wapper(x_emb, model_size)
    bw_hiddens, bw_hiddens_ori = encoder_wapper(x_emb_r, model_size)

    num_layers = len(fw_hiddens_ori)
    token_embeddings = fluid.layers.concat(input=[x_emb, x_emb], axis=1)
    concate_embeddings = [token_embeddings]

    for index in range(num_layers):
        embedding = fluid.layers.concat(
            input=[fw_hiddens_ori[index], bw_hiddens_ori[index]], axis=1)
        embedding = dropout(embedding)
        concate_embeddings.append(embedding)
    weighted_emb = weight_layers(concate_embeddings)
    return weighted_emb


def encoder_wapper(x_emb, emb_size, init_hidden=None, init_cell=None):
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
        rnn_out, cell, input_proj = lstmp_encoder(rnn_input, model_size, h0, c0, emb_size)
        rnn_out_ori = rnn_out
        if i > 0:
            rnn_out = rnn_out + rnn_input
        rnn_outs.append(rnn_out)
        rnn_outs_ori.append(rnn_out_ori)
    return rnn_outs, rnn_outs_ori


def lstmp_encoder(input_seq, gate_size, h_0, c_0, proj_size):
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
        proj_clip=proj_clip,
        cell_clip=cell_clip,
        proj_activation='identity')
    return hidden, cell, input_proj


def weight_layers(lm_embeddings):
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


def dropout(input, name):
    return fluid.layers.dropout(input, dropout_prob=dropout_rate, is_test=False, name=name)


def bigru_layer(input_feature):
    """
    define the bidirectional gru layer
    """
    pre_gru = fluid.layers.fc(
        input=input_feature,
        size=model_size * 3,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-init_bound, high=init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)))
    gru = fluid.layers.dynamic_gru(
        input=pre_gru,
        size=model_size,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-init_bound, high=init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)))

    pre_gru_r = fluid.layers.fc(
        input=input_feature,
        size=model_size * 3,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-init_bound, high=init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)))
    gru_r = fluid.layers.dynamic_gru(
        input=pre_gru_r,
        size=model_size,
        is_reverse=True,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-init_bound, high=init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)))

    bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
    return bi_merge
