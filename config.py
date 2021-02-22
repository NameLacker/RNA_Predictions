"""
配置文件
"""


class RNA_Config:
    # TODO: 最优损失  0.068
    def __init__(self):
        # ============================ 训练参数 ============================
        self.buf_size = 1000  # 缓冲区保存的数据个数
        self.batch_size = 1  # 批大小
        self.epochs = 12  # 一共训练多少个轮次
        self.val_batch = 475  # 多少个batch做一次验证
        self.best_dev_loss = 0.08  # 保存模型参数所需最高损失

        self.use_gpu = True  # 是否使用gpu
        self.add = 0  # 是否添加增广数据: 0--原数据, 1--预处理数据, 2--两者共用

        self.continue_train = True  # TODO: 是否加载前一次训练参数

        self.START = "<START>"  # 数据读取器相关参数
        self.STOP = "<STOP>"  # 数据读取器相关参数
        self.UNK = "<UNK>"  # 数据读取器相关参数

        # ========================== 网络模型参数 ===========================
        self.dmodel = 256  # embedding数据维度 TODO: 最优配置 128
        self.layers = 6  # lstm层数 TODO: 最优配置 6
        self.dropout = 0.15  # 模型参数丢弃概率

        # Elmo相关配置
        self.use_elmo = False  # 是否使用Elmo网络
        self.pre_training = False  # 是否固定Elmo预训练模型参数
        self.use_bigru = False  # 是否使用GRU
        self.cell_clip = 3.0
        self.proj_clip = 3.0
        self.init_bound = 0.1
        self.bigru_num = 2

        # ========================== 数据文件保存 ============================
        self.train_dataset = "./data/train.txt"  # 训练文件
        self.train_dataset_other = "./data/other_train.txt"  # 增广的训练文件
        self.dev_dataset = "./data/dev.txt"  # 验证文件
        self.test_dataset = "./data/test_nolabel.txt"  # 测试文件

        self.test = "./data/test.txt"  # 自己生成的带标签测试集
        self.train_log = "./log/train"  # visualdl格式log保存路径
        self.result = "./result/prediction"  # 测试结果保存文件夹
        self.params_dirname = "./inference_model"  # 模型文件存放文件夹

        self.test_dirname = "./inference_model/1613957122"  # 测试所用模型参数存放路径

        # ======================== 学习率动态调整策略 =========================
        self.beta1 = 0.9  # 梯度下降所需参数1
        self.beta2 = 0.999  # 梯度下降所需参数2
        self.epsilon = 1e-08  # 梯度下降所需参数3

        self.learn_rate = 0.0001  # 初始学习率
        self.each_step = 475  # 每隔多少步调整学习率
        self.boundaries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 经过多少个epoch降低学习率
        self.values = [1., 0.66, 0.33, 0.1, 0.066, 0.033, 0.01, 0.005, 0.001, 0.0001, 0.00001]  # 不同阶段学习率
