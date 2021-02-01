from __future__ import print_function
import paddle
import os
import time
import logging
from visualdl import LogWriter

from utils.inference_utils import *
from utils.reader import train_reader, val_reader
from net.network import *
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


def train_test(program, reader, feed_order, avg_cost, place):
    """
    计算训练中模型在test数据集上的结果
    :param feed_order:
    :param program:
    :param reader:
    :param avg_cost:
    :param accuracy:
    :param place:
    :return:
    """
    logger.info("Test Model...")
    feed_var_list = [program.global_block().var(var_name) for var_name in feed_order]

    feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    res_lose = 0.
    count = 0
    for test_data in reader():
        count += 1
        avg_cost_np = test_exe.run(
            program=program,
            feed=feeder_test.feed(test_data),
            fetch_list=[avg_cost])
        res_lose += avg_cost_np[0][0]
    res_cost = res_lose / 10
    return res_cost


def train():
    init_log_config()
    # 设置训练环境
    place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()

    # ============================ 构造数据读取器 ============================
    train_readers = paddle.batch(
        paddle.reader.shuffle(train_reader(), collocations.batch_size),
        batch_size=collocations.batch_size)
    test_readers = paddle.batch(
        paddle.reader.shuffle(val_reader(), collocations.batch_size),
        batch_size=collocations.batch_size)
    # ============================ 构造训练程序 ============================
    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    prediction = inference_program()
    avg_cost = train_program(prediction)

    sgd_optimizer, learn_rate = optimizer_func()
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    # ============================ 提供数据并构建主训练循环 ============================
    # 指定目录路径以保存参数
    params_dirname = collocations.params_dirname
    if not os.path.exists(params_dirname):
        os.makedirs(params_dirname)

    feed_order = ['rna', 'label', 'score']

    # 启动上下文构建的训练器
    feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
    exe.run(star_program)

    if collocations.continue_train:
        # 加载上一次训练模型
        logger.info("Loading model......")
        fluid.io.load_persistables(executor=exe,
                                   dirname=params_dirname,
                                   main_program=main_program,
                                   filename="persistables")
    log_name = str(int(time.time()))
    log_writer = LogWriter("./log/train" + log_name)
    # 训练主循环
    train_iters = 0
    fetch_list = [avg_cost.name, learn_rate]
    for epoch_id in range(collocations.epochs):
        for step_id, data in enumerate(train_readers()):
            # 运行训练器
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=fetch_list)
            cost, learn_rate = metrics[0][0], metrics[1][0]
            # 测试结果
            train_iters += 1
            log_writer.add_scalar(tag='train/loss', step=train_iters, value=cost)
            logger.info("Epoch: {}, Step: {}, Loss: {:.6}, Learn_rate: {:.7}".
                  format(epoch_id, step_id, cost, learn_rate))

        '''
        avg_cost_test = train_test(test_program, test_readers, feed_order, avg_cost, place)
        log_writer.add_scalar(tag='test/loss', step=epoch_id + 1, value=avg_cost_test)
        logger.info('Epoch {}, Test Loss {}'.format(epoch_id, avg_cost_test))
        '''
        logger.info("Save medol...")
        fluid.io.save_persistables(executor=exe, dirname=params_dirname,
                                   main_program=main_program, filename="persistables")
        fluid.io.save_inference_model(params_dirname, ["rna"], prediction, exe,
                                      params_filename="per_model", model_filename="__model__")


if __name__ == '__main__':
    train()
