

class Config:
    def __init__(self):
        self.batch_size = 128  # 批大小
        self.epochs = 50  # 一共训练多少个轮次
        self.use_gpu = True  # 是否使用gpu
        self.continue_train = True  # 是否加载前一次训练参数

        self.class_dim = 2  # 情感分类的类别数
        self.emb_dim = 128  # 词向量的维度
        self.hid_dim = 512  # 隐藏层的维度
        self.stacked_num = 9  # LSTM双向栈的层数

        self.max_acc = 0.  # 保存要求最低准确率

        self.params_dirname = "./inference_model"  # 模型文件存放文件夹

        self.learn_rate = 0.001  # 初始学习率
        self.boundaries = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # 经过多少个epoch降低学习率
        self.values = [1., 0.66, 0.33, 0.1, 0.066, 0.033, 0.01, 0.005, 0.001, 0.0001, 0.00001]  # 不同阶段学习率


class RNA_Config:
    def __init__(self):
        self.stacked_num = 15  # LSTM双向栈的层数
        self.batch_size = 32  # 批大小
        self.epochs = 50  # 一共训练多少个轮次
        self.use_gpu = True  # 是否使用gpu
        self.continue_train = False  # 是否加载前一次训练参数

        self.max_size = 20  # 输入数据统一尺寸
        self.class_dim = 500  # 输出尺寸
        self.emb_dim = 128  # 词向量的维度
        self.hid_dim = 512  # 隐藏层的维度

        self.max_acc = 0.  # 保存要求最低准确率

        self.params_dirname = "./inference_model"  # 模型文件存放文件夹
        self.train_dataset = "./data/rna/train.txt"
        self.dev_dataset = "./data/rna/dev.txt"
        self.test_dataset = "./data/rna/dev.txt"

        self.learn_rate = 0.001  # 初始学习率
        self.boundaries = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # 经过多少个epoch降低学习率
        self.values = [1., 0.66, 0.33, 0.1, 0.066, 0.033, 0.01, 0.005, 0.001, 0.0001, 0.00001]  # 不同阶段学习率

        self.input = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
        self.label = {'(': 1, ')': 2, '.': 3}
