from __future__ import print_function
import paddle
import os
import time
from visualdl import LogWriter

from utils.inference_utils import *
from net.network import *
from config import Config

collocations = Config()


def train_test(program, reader, feed_order, avg_cost, accuracy, place):
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
    print("Test Model...")
    count = 0
    feed_var_list = [program.global_block().var(var_name) for var_name in feed_order]

    feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, accuracy]) * [0]
    for test_data in reader():
        avg_cost_np = test_exe.run(
            program=program,
            feed=feeder_test.feed(test_data),
            fetch_list=[avg_cost, accuracy])
        accumulated = [
            x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
        ]
        count += 1
    return [x / count for x in accumulated]


def train():
    # 设置训练环境
    place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()

    # ============================ 构造数据读取器 ============================
    print("Loading IMDB word dict...")
    word_dict = paddle.dataset.imdb.word_dict()
    print("Reading training data...")
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=collocations.batch_size)
    print("Reading testing data...")
    test_reader = paddle.batch(paddle.dataset.imdb.test(word_dict), batch_size=collocations.batch_size)

    # ============================ 构造训练程序 ============================
    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    prediction = inference_program(word_dict)
    train_func_outputs, accurary = train_program(prediction)
    avg_cost = train_func_outputs[0]

    test_program = main_program.clone(for_test=True)

    sgd_optimizer, learn_rate = optimizer_func()
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    # ============================ 提供数据并构建主训练循环 ============================
    # 指定目录路径以保存参数
    params_dirname = collocations.params_dirname
    if not os.path.exists(params_dirname):
        os.makedirs(params_dirname)

    feed_order = ['words', 'label']

    # 启动上下文构建的训练器
    feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
    exe.run(star_program)

    if collocations.continue_train:
        # 加载上一次训练模型
        print("Loading model......")
        fluid.io.load_persistables(executor=exe,
                                   dirname=params_dirname,
                                   main_program=main_program,
                                   filename="persistables")

    log_name = str(int(time.time()))
    log_writer = LogWriter("./log/train" + log_name)
    # 训练主循环
    train_iters = 0
    max_acc = collocations.max_acc
    fetch_list = [var.name for var in train_func_outputs]
    fetch_list.append(learn_rate)
    for epoch_id in range(collocations.epochs):
        for step_id, data in enumerate(train_reader()):
            print(data)
            # 运行训练器
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=fetch_list)
            acc, cost, learn_rate = metrics[1][0], metrics[0][0], metrics[2][0]
            # 测试结果
            train_iters += 1
            log_writer.add_scalar(tag='train/acc', step=train_iters, value=acc)
            log_writer.add_scalar(tag='train/loss', step=train_iters, value=cost)
            print("Epoch: {}, Step: {}, Accurary: {:.6}, Loss: {:.6}, Learn_rate: {:.7}".
                  format(epoch_id, step_id, acc, cost, learn_rate))

        # 验证程序
        avg_cost_test, acc_test = train_test(test_program, test_reader, feed_order, avg_cost, accurary, place)
        log_writer.add_scalar(tag='test/acc', step=epoch_id+1, value=acc_test)
        log_writer.add_scalar(tag='test/loss', step=epoch_id+1, value=avg_cost_test)
        print('Epoch {}, Acc {}, Test Loss {}'.format(epoch_id, acc_test, avg_cost_test))

        if acc_test > max_acc:
            max_acc = acc_test
            print("Save model...")
            fluid.io.save_persistables(executor=exe, dirname=params_dirname,
                                       main_program=main_program, filename="persistables")
            fluid.io.save_inference_model(params_dirname, ["words"], prediction, exe,
                                          params_filename="per_model", model_filename="__model__")  # 保存模型


if __name__ == '__main__':
    train()
