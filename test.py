from __future__ import print_function
import paddle
import numpy as np
import paddle.fluid as fluid

from config import Config

collocations = Config()


# 构建预测器
place = fluid.CUDAPlace(0) if collocations.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

word_dict = paddle.dataset.imdb.word_dict()
[inferencer, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(collocations.params_dirname, exe, params_filename="per_model")


def infer(review_str):
    # 生成测试用输入数据
    review = review_str.split()

    UNK = word_dict['<unk>']
    lod = [[np.int64(word_dict.get(words, UNK)) for words in review]]

    base_shape = [[len(c) for c in lod]]

    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
    assert feed_target_names[0] == "words"
    results = exe.run(inferencer,
                      feed={feed_target_names[0]: tensor_words},
                      fetch_list=fetch_targets,
                      return_numpy=False)
    np_data = np.array(results[0])
    for i, r in enumerate(np_data):
        print("Predict probability of ", r[0], " to be positive and ", r[1],
              " to be negative for review \'", review_str, "\'")


if __name__ == '__main__':
    while True:
        input_words = input("Please input your criticism: ")
        infer(input_words)
