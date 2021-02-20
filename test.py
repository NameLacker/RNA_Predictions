from __future__ import print_function
import numpy as np
import os
import zipfile
import time
import paddle.fluid as fluid

from config import RNA_Config
from net.network import Network
from utils.process import process_vocabulary
from utils.reader import load_train_data, load_test_data, reader_creator

collocations = RNA_Config()


def run_test(args):
    """
    执行预测
    :param args:
    :return:
    """
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    print("Loading data...")
    train_data, val_data = load_train_data()
    test_data = load_test_data()

    print("Loading model...")
    seq_vocab, bracket_vocab = process_vocabulary(train_data, quiet=True)
    network = Network(
        seq_vocab,
        bracket_vocab,
        dmodel=args.dmodel,
        layers=args.layers,
        dropout=0,
    )

    exe = fluid.Executor(place)
    fluid.io.load_inference_model(args.test_dirname, exe, params_filename="per_model")
    test_reader = fluid.io.batch(
        reader_creator(test_data, seq_vocab, bracket_vocab, test=True),
        batch_size=args.batch_size)

    seq = fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
    dot = fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
    predictions = network(seq, dot)

    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)
    test_feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot])

    test_results = []
    for id, data in enumerate(test_reader()):
        pred, = exe.run(test_program,
                        feed=test_feeder.feed(data),
                        fetch_list=[predictions.name],
                        return_numpy=False
                        )
        pred = list(np.array(pred))
        test_results.append(pred)
        # 打印预测信息
        print("Prediction RNA: {}\n"
              "RNA Sequence: {}\n"
              "RNA Structure: {}\n"
              "RNA Probability: {}\n"
              "RNA Length: {}\n"
              .format(test_data[id]["id"].split()[1], test_data[id]["sequence"],
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


if __name__ == '__main__':
    collocations = RNA_Config()  # 获取配置信息
    results = run_test(collocations)
    save_results(results)
    print("Zip result...")
    time.sleep(5)
    writeAllFileToZip("./result", "predict.files.zip")
    print("Success zip result!")
