# 螺旋桨RNA结构预测竞赛：Unpaired Probability Prediction  
***
成员有： 舒涛，姚智超，朱满琴，邱凯 and so on...  
内容有待开发，前路一片光明。  
革命尚未成功，同志仍需努力。  
加油，奥利给！！！

## 项目所需依赖库
***
运行`pip install -r requestments -i https://pypi.douban.com/simple`下载安装所需依赖  
项目目录树：  
.  
├── config.py           &#8195;    // 配置文件  
├── data  
│   ├── data_explanation.txt   &#8195;             // 数据说明文件  
│   ├── dev.txt                &#8195;             // 验证数据集文件  
│   ├── README.md              &#8195;             // 数据说明.md  
│   ├── test_nolabel.txt       &#8195;             // A榜测试数据  
│   └── train.txt              &#8195;             // 训练数据集文件  
├── demo.py            &#8195;      // 演示脚本  
├── inference_model  
│   ├── __model__      &#8195;      // 模型结构图（visualdl打开）  
│   ├── per_model      &#8195;     // 预测模型参数  
│   └── persistables   &#8195;     // 训练模型参数  
├── logs  
│   └── train_1612239291.log       &#8195;         // 日志文件  
├── net  
│   ├── __init__.py  
│   ├── network.py          &#8195;     // 网络模型代码  
├── README.md               &#8195;     // 项目说明  
├── requestments.txt        &#8195;     // 配置库  
├── result  
│   ├── prediction          &#8195;     // 预测结果存放路径  
│   │   ├── 1.predict.txt   &#8195;     // 预测结果1  
│   │   ├── 2.predict.txt   &#8195;     // 预测结果2  
            ...
            ...
            ...
│   │   ├── n.predict.txt   &#8195;     // 预测结果n  
│   ├── __model__.svg       &#8195;     // 模型结构图  
│   ├── predict.file.zip    &#8195;     // 待提交压缩文件  
└── utils  
    ├── __init__.py  
    ├── process.py          &#8195;     // 创建词汇表  
    ├── reader.py           &#8195;     // 数据读入程序  
    └── vocabulary.py       &#8195;     // 数据格式化类  
├── test.py                 &#8195;     // 验证程序  
├── train.py                &#8195;     // 训练程序  


## 测试项目  
***
* 运行`python train.py`开始训练，训练参数会保存在`./inference_model`目录下  
* 运行`visualdl --logdir ./log post 8040`，再在浏览器打开`http://localhost:8040/`可以查看训练进度  
* 运行`python test.py`产生测试结果，测试结果保存在`./result`目录下  

## 模型
***
主要结构采用bidirectional-LSTM（双向LSTM网络结构），and so on......  
可以尝试从以下几方面来进行调优:（此处调用官方提示说明）  
* 输入数据预处理，提取更多feature
    * 基线模型使用LinearFold预测的RNA二级结构作为辅助feature。选手可以尝试增加更多的辅助feature，如：使用其他二级结构预测软件（如Vienna RNAfold, RNAstructure, CONTRAfold等）生成新的二级结构feature。
* 更复杂的Embedding形式
* 可以尝试在Embedding层使用Elmo, Bert等预训练模型
* 优化网络结构和参数
    * 隐层大小选择 - 宽度和层数
    * 尝试复杂网络构建
    * 尝试正则化、dropout等方式避免过拟合
    * 选择学习率等超参数
    * 选择合适的损失函数
    * ~~尝试不同的优化器~~

## 不同参数下的验证结果
***
<table>
    <thead>
        <tr>
            <th>Embedding数据维度</th>
            <th>Lstm 层数</th>
            <th>Dropout概率</th>
            <th>Loss</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>128</th>
            <th>8</th>
            <th>0.15</th>
            <th>0.068</th>
        </tr>
        <tr>
            <th>256</th>
            <th>8</th>
            <th>0.15</th>
            <th>0.078697</th>
        </tr>
        <tr>
            <th>512</th>
            <th>16</th>
            <th>0.15</th>
            <th>None</th>
        </tr>
    </tbody>
</table>


## 当前模型结构
***  
![model](https://github.com/NameLacker/RNA_Prediction/blob/master/result/__model__.svg)