# 螺旋桨RNA结构预测竞赛：Unpaired Probability Prediction  
***
成员有： 舒涛，姚智超，朱满琴，邱凯，叶永志 and so on...  
内容有待开发，前路一片光明。  
革命尚未成功，同志仍需努力。  
加油，奥利给！！！

## 项目目录树
***
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
│   │   ├──&#8195;  ...  
│   │   ├──&#8195;  ...  
│   │   ├──&#8195;  ...   
│   └── ├──&#8195;  ...    &#8195;     // 训练模型参数  
├── logs  
│   └── train_1612239291.log       &#8195;         // 日志文件  
├── max_models         &#8195;    // 最优模型存放目录  
│   │   ├──&#8195;  ...  
│   │   ├──&#8195;  ...  
│   │   ├──&#8195;  ...   
│   └── 4.635          &#8195;         // 最优模型  
├── net  
│   ├── network.py          &#8195;     // 网络模型代码  
│   └── bilm.py             &#8195;     // Elmo结构  
├── README.md               &#8195;     // 项目说明  
├── requestments.txt        &#8195;     // 配置库  
├── result  
│   ├── prediction          &#8195;     // 预测结果存放路径  
│   │   ├── 1.predict.txt   &#8195;     // 预测结果1  
│   │   ├── 2.predict.txt   &#8195;     // 预测结果2  
│   │   ├──&#8195;  ...  
│   │   ├──&#8195;  ...  
│   │   ├──&#8195;  ...  
│   │   ├── n.predict.txt   &#8195;     // 预测结果n  
│   ├── __model__.svg       &#8195;     // 模型结构图  
│   ├── predict.file.zip    &#8195;     // 待提交压缩文件  
└── utils  
    ├── process.py          &#8195;     // 创建词汇表  
    ├── reader.py           &#8195;     // 数据读入程序  
    ├── utils.py            &#8195;     // 数据预处理  
    └── vocabulary.py       &#8195;     // 数据格式化类  
├── test.py                 &#8195;     // 验证程序  
├── train.py                &#8195;     // 训练程序  


## 项目依赖下载，模型训练、测试  
***
* 运行`pip install -r requestments -i https://pypi.douban.com/simple`下载安装所需依赖  
* 运行`python train.py`开始训练，训练参数会保存在`./inference_model`目录下  
* 运行`visualdl --logdir ./log post 8040`，再在浏览器打开`http://localhost:8040/`可以查看训练进度  
* 运行`python test.py`产生测试结果（可设置读取模型的目录参数`--param_path=参数目录`，各个模型参数全在`./max_models`目录下，也可以读取`./inference`下的模型参数，如不设置此参数，则默认值为`./max_models/3.739B`），测试结果保存在`./result/prediction`目录下，同时在`./result`目录下生成`predict.files.zip`作为提交文件  

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
* A榜：  
<table align="center">
    <thead>
        <tr>
            <th>序号</th>
            <th>Embedding数据维度</th>
            <th>Lstm 层数</th>
            <th>Loss</th>
            <th>Step</th>
            <th>Score</th>
            <th>rmsd_avg</th>
            <th>rmsd_std</th>
            <th>时间</th>
            <th>备注</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>128</td>
            <td>8</td>
            <td>0.068</td>
            <td>None</td>
            <td>4.296</td>
            <td>0.233</td>
            <td>0.041</td>
            <td>2021-02-19 09:01</td>
            <td>None</td>
        </tr>
        <tr>
            <td>2</td>
            <td>128</td>
            <td>6</td>
            <td>0.075</td>
            <td>4270</td>
            <td>4.534</td>
            <td>0.221</td>
            <td>0.046</td>
            <td>2021-02-19 09:42</td>
            <td>None</td>
        </tr>
        <tr>
            <td>3</td>
            <td>128</td>
            <td>6</td>
            <td>0.076</td>
            <td>4750</td>
            <td>4.562</td>
            <td>0.219</td>
            <td>0.048</td>
            <td>2021-02-19 10:06</td>
            <td>None</td>
        </tr>
        <tr>
            <td>4</td>
            <td>256</td>
            <td>6</td>
            <td>0.074</td>
            <td>8550</td>
            <td>4.635</td>
            <td>0.216</td>
            <td>0.05</td>
            <td>2021-02-20 09:05</td>
            <td>None</td>
        </tr>
        <tr>
            <td>5</td>
            <td>256</td>
            <td>6</td>
            <td>0.071</td>
            <td>4270</td>
            <td>4.595</td>
            <td>0.218</td>
            <td>0.049</td>
            <td>2021-02-22 09:45</td>
            <td>续4的模型参数继续训练</td>
        </tr>
        <tr>
            <td>6</td>
            <td>256</td>
            <td>6</td>
            <td>0.077</td>
            <td>1420</td>
            <td>4.587</td>
            <td>0.218</td>
            <td>0.054</td>
            <td>2021-03-01 10:15</td>
            <td>续4的模型参数继续训练</td>
        </tr>
        <tr>
            <td>7</td>
            <td>256</td>
            <td>6</td>
            <td>0.072</td>
            <td>475</td>
            <td>4.632</td>
            <td>0.216</td>
            <td>0.05</td>
            <td>2021-03-04 10:56:23</td>
            <td>续4的模型参数继续训练，使用了增广数据</td>
        </tr>
    </tbody>
</table>


## 模型训练过程综述
***  
* 在模型上的深度改进上，发现在Baseline基线模型上增加LSTM层数会导致损失剧增，所以选择减小模型参数，经过大量的实验，发现当`LSTM layers=6`时在A榜可达到最佳评分。  
* 在模型输入尺寸上，增大或减小`embedding`的数据维度也会导致损失的变化，当其为`256`时亦可使得A榜测试达到最佳。
* 在`embedding`模型上的改进，我们试着引进`Elem`、`GRU`网络结构，在实际训练中，损失最终会稳定在和未添此trip大致一样的数值，但当提交此模型产生的预测结果到A榜后会导致分数骤降，所以在之后的训练实验中未采取此trip。　　
* 在数据的预处理中，我们试着用官方工具`PaddleHelix`针对数据集的碱基序列产生新的二级结构，此trip帮助我们在A榜上取得了最佳的成绩，但在B榜上的表现并不如人意，之后我们又采用同源序列的方法对数据集预处理，此trip使得我们在B榜上取得了最佳的成绩。　　
* 模型超参数的改进，`learning_rate`采取阶梯下降的方式，优化方法试过几乎全部的优化器，还是`Baseline`的优化器取得的成绩最高。增大训练的`Batch size`也没有很好的表现。
* 此外，由于官方不提供`RMSD_STD`的计算方案，所以我们在对A榜的所有提交数据的分析下，自创了`RMST_STD`的计算方案，并应用于训练时的验证上。