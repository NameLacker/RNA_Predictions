from __future__ import print_function
import numpy as np
import os
import paddle.fluid as fluid

from config import RNA_Config
from net.network import Network
from utils.process import process_vocabulary
from utils.reader import load_train_data, load_test_data, reader_creator


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
    fluid.io.load_inference_model(args.params_dirname, exe, params_filename="per_model")
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
        print("Prediction RNA: {}\n"
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
    print("Start save results......")
    for id, result in enumerate(results):
        id += 1
        save_file = os.path.join(collocations.result, str(id) + ".predict.txt")
        with open(save_file, 'w') as f:
            for res in result:
                f.write(str(res) + "\n")
    print("Success save results......")


if __name__ == '__main__':
    collocations = RNA_Config()
    results = run_test(collocations)
    save_results(results)
