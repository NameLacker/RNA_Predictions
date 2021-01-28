# 螺旋桨RNA结构预测竞赛：Unpaired Probability Prediction  
***
内容有待开发，前路一片光明。  
革命尚未成功，同志仍需努力。  
加油，奥利给！！！

## 项目所需依赖库
***
运行`pip install -r requestments -i https://pypi.douban.com/simple`下载安装所需依赖

## 测试项目  
***
* 运行`python train.py`开始训练，训练结果参数会保存在`./inference_model`目录下  
* 运行`visualdl --logdir ./log post 8040`，再在浏览器打开`http://localhost:8040/`可以查看训练进度  
* 运行`python test.py`测试训练结果，此测试是针对影评的情感分类，共两个分类--积极和消极  
* 测试模型图片  
![Image](https://github.com/NameLacker/RNA_Prediction/blob/main/result/__model__.svg)

## 模型
***
主要结构采用bidirectional-LSTM（双向LSTM网络结构），and so on......  