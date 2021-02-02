"""
配置文件
"""


class RNA_Config:
    def __init__(self):
        # ============================ 训练参数 ============================
        self.buf_size = 500  # 缓冲区保存的数据个数
        self.batch_size = 1  # 批大小
        self.epochs = 30  # 一共训练多少个轮次
        self.use_gpu = True  # 是否使用gpu
        self.continue_train = True  # 是否加载前一次训练参数
        self.best_dev_loss = 100.  # 最低保存模型参数所需损失
        self.START = "<START>"  # 数据读取器相关参数
        self.STOP = "<STOP>"  # 数据读取器相关参数
        self.UNK = "<UNK>"  # 数据读取器相关参数

        # ============================ 网络模型参数 ============================
        self.dmodel = 128  # embedding数据维度
        self.layers = 8  # lstm层数
        self.dropout = 0.15  # 模型参数丢弃概率

        # ============================ 数据文件保存 ============================
        self.params_dirname = "./inference_model"  # 模型文件存放文件夹
        self.train_dataset = "./data/train.txt"  # 训练文件
        self.train_dataset_other = "./data/other_train.txt"
        self.dev_dataset = "./data/dev.txt"  # 验证文件
        self.test_dataset = "./data/test_nolabel.txt"  # 测试文件
        self.test = "./data/test.txt"  # 自己生成的带标签测试集
        self.train_log = "./log/train"  # visualdl格式log保存路径
        self.result = "./result/prediction"  # 测试结果保存文件夹

        # ============================ 学习率动态调整策略 ============================
        self.beta1 = 0.9  # 梯度下降所需参数1
        self.beta2 = 0.999  # 梯度下降所需参数2
        self.epsilon = 1e-08  # 梯度下降所需参数3
        self.learn_rate = 0.001  # 初始学习率
        self.each_step = 5000  # 每隔多少步调整学习率
        self.boundaries = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # 经过多少个epoch降低学习率
        self.values = [1., 0.66, 0.33, 0.1, 0.066, 0.033, 0.01, 0.005, 0.001, 0.0001, 0.00001]  # 不同阶段学习率



