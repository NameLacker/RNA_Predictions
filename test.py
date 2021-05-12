from __future__ import print_function
import numpy as np
import os
import zipfile
import time
import argparse
import paddle.fluid as fluid

from config import RNA_Config
from net.network import Network
from utils.process import process_vocabulary
from utils.reader import load_train_data, reader_creator, load_test_A, load_test_B

collocations = RNA_Config()

place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

assert len(os.listdir(collocations.freeze_dirname)) > 0, "请先执行固化程序"

# 读取固化模型参数
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
    dirname=collocations.freeze_dirname, executor=exe, params_filename="pre_model")


def run_test(args):
    """
    执行预测
    :param args:
    :return:
    """
    print("Loading data...")
    train_data, val_data = load_train_data()
    # test_data = load_test_A()
    test_data = load_test_B()  # todo: B榜测试数据

    print("Loading model...")
    seq_vocab, bracket_vocab = process_vocabulary(train_data, quiet=True)

    test_reader = fluid.io.batch(
        reader_creator(test_data, seq_vocab, bracket_vocab, test=True),
        batch_size=args.test_size)

    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)

    test_feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot])

    test_results = []
    for id, data in enumerate(test_reader()):
        pred, = exe.run(inference_program,
                        feed=test_feeder.feed(data),
                        fetch_list=fetch_targets,
                        return_numpy=False)
        pred = list(np.array(pred))
        test_results.append(pred)
        # 打印预测信息
        print("{}\n"
              "RNA Sequence: {}\n"
              "RNA Structure: {}\n"
              "RNA Probability: {}\n"
              "RNA Length: {}\n"
              .format(test_data[id]["id"], test_data[id]["sequence"],
                      test_data[id]["structure"], pred, len(pred)))
    return test_results


def save_results(results):
    """
    保存结果
    :param results:
    :return:
    """
    try:
        # 删除旧结果
        print("Delete old file......")
        res_list = os.listdir(collocations.result)
        for file in res_list:
            file_path = os.path.join(collocations.result, file)
            os.remove(file_path)
        time.sleep(1)
    except Exception as e:
        pass
    if not os.path.exists(collocations.result):
        os.makedirs("./result/prediction")
    print("Start save results......")
    for id, result in enumerate(results):
        id += 1
        save_file = os.path.join(collocations.result, str(id) + ".predict.txt")
        with open(save_file, 'w') as f:
            for res in result:
                f.write(str(res) + "\n")
    print("Success save results......")


def writeAllFileToZip(absDir, zipFile):
    """
    定义一个函数，递归读取absDir文件夹中所有文件，并塞进zipFile文件中。参数absDir表示文件夹的绝对路径。
    :param absDir:
    :param zipFile:
    :return:
    """
    os.chdir(absDir)
    absDir = "./prediction"
    zipFile = zipfile.ZipFile(zipFile, 'w', zipfile.ZIP_DEFLATED)

    for f in os.listdir(absDir):
        absFile = os.path.join(absDir, f)  # 子文件的绝对路径
        zipFile.write(absFile)  # 逐文件压缩


def freeze_model(path, save_dir):
    """
    固化模型参数
    :param path: 训练模型参数
    :param save_dir: 固化模型存放目录
    :return:
    """
    # 设置训练环境
    place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 读取数据
    train_data, val_data = load_train_data()
    # 构建数据字典
    seq_vocab, bracket_vocab = process_vocabulary(train_data)
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
    # 前向传播
    predictions = network(seq, dot)

    freeze_program = fluid.default_main_program()
    fluid.io.load_persistables(exe, path, freeze_program)
    freeze_program = freeze_program.clone(for_test=True)

    fluid.io.save_inference_model(save_dir, ["seq", "dot"], predictions, exe, freeze_program,
                                  model_filename="__model__", params_filename="pre_model")
    print("freeze out: {0}, pred layout: {1}".format(collocations.freeze_dirname, predictions))
    print("固化模型成功!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--param_path', type=str, default="./max_models/3.739B")
    path = parser.parse_args().param_path
    # 固化模型
    print("读取参数：{}".format(path))
    freeze_model(path, "freeze_model/")

    time.sleep(3)
    # 开始预测
    collocations = RNA_Config()  # 获取配置信息
    results = run_test(collocations)  # 预测产生结果
    save_results(results)  # 保存结果
    print("Zip result...")
    time.sleep(3)
    writeAllFileToZip("./result", "predict.files.zip")  # 压缩结果进`predict.files.zip`文件
    print("Success zip result!")
