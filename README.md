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
├── config.py  
├── data  
│   ├── data_explanation.txt  
│   ├── dev.txt  
│   ├── README.md  
│   ├── test_nolabel.txt  
│   └── train.txt  
├── demo.py  
├── inference_model  
│   ├── __model__  
│   ├── per_model  
│   └── persistables  
├── logs  
│   └── train_1612239291.log  
├── net  
│   ├── __init__.py  
│   ├── network.py  
├── README.md  
├── requestments.txt  
├── result  
├── test.py  
├── train.py  
├── tree.txt  
└── utils  
    ├── __init__.py  
    ├── process.py  
    ├── reader.py  
    └── vocabulary.py  

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
            <th>0.0699</th>
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