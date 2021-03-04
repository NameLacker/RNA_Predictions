from __future__ import print_function

import math
import os
import sys
import time
import logging
import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter

from utils.process import process_vocabulary, operator_rmsd_avg
from utils.reader import load_train_data, reader_creator
from net import optimization
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

    # =============================== 构造训练程序 =============================
    # 设置训练环境
    place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    # ============================ 解析数据并构建数据读取器 =======================
    # 读取数据
    train_data, val_data = load_train_data()
    print("Size of Train Dataset: {}".format(len(train_data)))
    # 构建数据字典
    seq_vocab, bracket_vocab = process_vocabulary(train_data)
    # 训练数据读取
    train_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(train_data, seq_vocab, bracket_vocab), buf_size=collocations.buf_size),
        batch_size=collocations.batch_size)
    # 验证数据读取
    val_reader = fluid.io.batch(
        fluid.io.shuffle(
            reader_creator(val_data, seq_vocab, bracket_vocab), buf_size=collocations.buf_size),
        batch_size=collocations.batch_size)

    # ============================ 构造数据容器和网络 ============================
    # 读取网络模型
    network = Network(
        seq_vocab,
        bracket_vocab,
        dmodel=collocations.dmodel,
        layers=collocations.layers,
        dropout=collocations.dropout,
    )
    # 构建数据容器
    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
    y = fluid.data(name="label", shape=[None], dtype="float32")
    # 前向传播
    predictions = network(seq, dot)
    # 交叉熵损失函数
    loss = fluid.layers.mse_loss(input=predictions, label=y)
    avg_loss = fluid.layers.mean(loss)

    # 复制测试程序
    test_program = main_program.clone(for_test=True)
    # 构造数据读取器
    feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot, y])
    # 构造直方图
    params = [param.name for param in fluid.default_main_program().all_parameters()]

    optimizer, learn_rate = optimization.optimizer()
    # 反向传播，计算梯度
    optimizer.minimize(avg_loss)
    exe.run(start_program)

    # ============================ 保存训练日志及模型参数 ==========================
    # 模型保存路径
    params_dirname = collocations.params_dirname

    # 训练日志保存
    log_name = str(int(time.time()))
    log_writer = LogWriter(collocations.train_log + log_name)
    train_iters = 0

    # ========================== 是否加载上一次训练模型参数 =========================
    if collocations.continue_train:
        logger.info("Loading model......")
        # 加载上一次训练模型参数
        fluid.io.load_persistables(executor=exe, dirname=params_dirname, main_program=main_program, filename="persistables")

    avg_batch_loss = 0.  # 最小loss
    t = 0.
    for epoch_id in range(collocations.epochs):
        # ============================ 构造数据读取器 =============================
        train_reader = fluid.io.batch(
            fluid.io.shuffle(
                reader_creator(train_data, seq_vocab, bracket_vocab), buf_size=collocations.buf_size),
            batch_size=collocations.batch_size)

        # =============================== 开始训练 ===============================
        for step_id, data in enumerate(train_reader()):
            # 运行训练器
            t1 = time.time()
            batch_loss, pred_values, learning_rate = exe.run(main_program, feed=feeder.feed(data),
                                                             fetch_list=[avg_loss.name, predictions.name, learn_rate],
                                                             return_numpy=False)
            t2 = time.time() - t1
            t += t2
            batch_loss = np.array(batch_loss)[0]
            learning_rate = np.array(learning_rate)[0]
            avg_batch_loss += batch_loss

            if math.isnan(float(batch_loss)):
                sys.exit("got NaN loss, training failed.")

            # =============================== 打印日志 ===============================
            train_iters += 1
            if train_iters % 10 == 0:
                batch_loss = avg_batch_loss / 10
                log_writer.add_scalar(tag='train/loss', step=train_iters, value=float(batch_loss))
                log_writer.add_scalar(tag='train/learning_rate', step=train_iters, value=float(learning_rate))
                logger.info("Epoch: {}, Step: {}, Loss: {:.8}, Learning_rate: {:.8}, Cost_time: {:.5}".
                            format(epoch_id, step_id+1, batch_loss, learning_rate, t))
                avg_batch_loss = 0.
                t = 0.

                for param in params:
                    values = fluid.global_scope().find_var(param).get_tensor()
                    log_writer.add_histogram(tag="train/{}".format(param), step=train_iters, values=values)

            # =============================== 验证程序 ===============================
            if train_iters % collocations.val_batch == 0:
                val_results = []
                preds = []
                labels = []
                for data in val_reader():
                    loss, pred = exe.run(test_program,
                                         feed=feeder.feed(data),
                                         fetch_list=[avg_loss.name, predictions.name],
                                         return_numpy=False)
                    loss = np.array(loss)
                    pred = list(np.array(pred))
                    label = list(data[0][2])

                    preds.append(pred)
                    labels.append(label)
                    val_results.append(loss[0])

                rmsd_avg, rmsd_std = operator_rmsd_avg(preds, labels)  # 计算验证集平均RMSD
                val_loss = sum(val_results) / len(val_results)
                log_writer.add_scalar(tag='test/loss', step=train_iters, value=val_loss)
                log_writer.add_scalar(tag='test/rmsd_avg', step=train_iters, value=rmsd_avg)
                logger.info("Epoch: {}, Test Loss: {}, Test Rmsd_avg: {:.8}, Test Rmsd_std: {:.8}"
                            .format(epoch_id, val_loss, rmsd_avg, rmsd_std))

                # =============================== 保存模型参数 ===============================
                if (rmsd_avg < collocations.best_dev_rmsd and val_loss < collocations.best_dev_loss) or (step_id == 0 and epoch_id > 0):
                    savename = "{}".format(int(time.time()))
                    savename = os.path.join(collocations.save_dirname, savename)
                    if not os.path.exists(savename):
                        os.makedirs(savename)
                    collocations.best_dev_rmsd = rmsd_avg
                    logger.info("Save medol...")
                    fluid.io.save_persistables(executor=exe, dirname=savename,
                                               main_program=main_program)
                    fluid.io.save_inference_model(savename, ['seq', 'dot'], [predictions], exe,
                                                  params_filename="per_model", model_filename="__model__")


if __name__ == '__main__':
    train()
