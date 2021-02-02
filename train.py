from __future__ import print_function

import math
import os
import sys
import time
import logging
import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter

from utils.process import process_vocabulary
from utils.reader import load_train_data, reader_creator
from net.network import Network
from config import RNA_Config

collocations = RNA_Config()

logger = None


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # os.getcwd(): 返回当前路径
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train_' + str(int(time.time())) + '.log')
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def train():
    init_log_config()  # 初始化日志

    # =============================== 构造训练程序 ==============================
    # 设置训练环境
    place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    # ============================ 解析数据并构建数据读取器 =======================
    train_data, val_data = load_train_data()
    seq_vocab, bracket_vocab = process_vocabulary(train_data)
    train_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(train_data, seq_vocab, bracket_vocab), buf_size=collocations.buf_size),
        batch_size=collocations.batch_size)
    val_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(val_data, seq_vocab, bracket_vocab), buf_size=collocations.buf_size),
        batch_size=collocations.batch_size)

    # ============================ 构造数据容器和网络 ============================
    network = Network(
        seq_vocab,
        bracket_vocab,
        dmodel=collocations.dmodel,
        layers=collocations.layers,
        dropout=collocations.dropout,
    )
    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
    y = fluid.data(name="label", shape=[None], dtype="float32")
    predictions = network(seq, dot)
    loss = fluid.layers.mse_loss(input=predictions, label=y)
    avg_loss = fluid.layers.mean(loss)

    test_program = main_program.clone(for_test=True)
    feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot, y])

    boundaries = [step * collocations.each_step for step in collocations.boundaries]
    values = [value * collocations.learn_rate for value in collocations.values]
    learn_rate = fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.Adam(
        learning_rate=learn_rate,
        beta1=collocations.beta1,
        beta2=collocations.beta2,
        epsilon=collocations.epsilon,
    )
    optimizer.minimize(avg_loss)
    exe.run(start_program)

    # ============================ 保存训练日志及模型参数 ============================
    params_dirname = collocations.params_dirname
    log_name = str(int(time.time()))
    log_writer = LogWriter(collocations.train_log + log_name)
    train_iters = 0

    # ============================ 是否加载上一次训练模型参数 ==========================
    if collocations.continue_train:
        # 加载上一次训练模型参数
        logger.info("Loading model......")
        fluid.io.load_persistables(executor=exe, dirname=params_dirname,
                                   main_program=main_program, filename="persistables")

    avg_batch_loss = 0.
    for epoch_id in range(collocations.epochs):
        # ============================== 构造数据读取器 ==============================
        train_reader = fluid.io.batch(
            fluid.io.shuffle(
                reader_creator(train_data, seq_vocab, bracket_vocab), buf_size=collocations.buf_size),
            batch_size=collocations.batch_size)

        # =============================== 开始训练 ===============================
        for step_id, data in enumerate(train_reader()):
            # 运行训练器
            batch_loss, pred_values, learning_rate = exe.run(main_program, feed=feeder.feed(data),
                                                             fetch_list=[avg_loss.name, predictions.name, learn_rate],
                                                             return_numpy=False)
            batch_loss = np.array(batch_loss)[0]
            learning_rate = np.array(learning_rate)[0]
            avg_batch_loss += batch_loss

            if math.isnan(float(batch_loss)):
                sys.exit("got NaN loss, training failed.")

            # =============================== 打印日志 ===============================
            train_iters += 1
            if train_iters % 20 == 0:
                batch_loss = avg_batch_loss / 20
                log_writer.add_scalar(tag='train/loss', step=train_iters, value=float(batch_loss))
                logger.info("Epoch: {}, Step: {}, Loss: {:.8}, Learning_rate: {:.8}".
                            format(epoch_id, step_id+1, batch_loss, learning_rate))
                avg_batch_loss = 0.

        # =============================== 验证程序 ===============================
        val_results = []
        for data in val_reader():
            loss, pred = exe.run(test_program,
                                 feed=feeder.feed(data),
                                 fetch_list=[avg_loss.name, predictions.name],
                                 return_numpy=False
                                 )
            loss = np.array(loss)
            val_results.append(loss[0])
        val_loss = sum(val_results) / len(val_results)
        log_writer.add_scalar(tag='test/loss', step=train_iters, value=val_loss)
        logger.info("Epoch: {}, Test Loss: {}".format(epoch_id, val_loss))

        # =============================== 保存模型参数 ===============================
        if val_loss <= collocations.best_dev_loss:
            collocations.best_dev_loss = val_loss
            logger.info("Save medol...")
            fluid.io.save_persistables(executor=exe, dirname=params_dirname,
                                       main_program=main_program, filename="persistables")
            fluid.io.save_inference_model(params_dirname, ['seq', 'dot'], [predictions], exe,
                                          params_filename="per_model", model_filename="__model__")


if __name__ == '__main__':
    train()
