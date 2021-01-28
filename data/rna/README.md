# About the DATA

train.txt and dev.txt contain the data as training set and validation set.   
train.txt contains 4,750 data points; dev.txt contains 250 data points.  
Each data point has four parts:  
* ID, e.g., ">id_1";
* RNA sequence;
* LinearFold predicted structure, where "." represents unpaired nucleotide, "(" and ")" represent base pairs;
* Output labels, i.e., the unpaired probability of each position

test_nolabel.txt contains the data as testing set of Leading Board A. 
It contains 444 data points; each data point has three parts:
* ID;
* RNA sequence;
* LinearFold predicted structure.

# 关于数据

train.txt和dev.txt包含作为训练集和验证集的数据。  
train.txt包含4,750个数据； dev.txt包含250个数据。  
每个数据点包含四个部分：  
* ID，例如“> id_1”；  
* RNA序列；  
* LinearFold预测结构，其中“.” 代表未配对的核苷酸，“（”和“）”代表碱基对；  
* 输出标签，即每个位置的未配对概率  

test_nolabel.txt包含数据作为排行榜A的测试集。  
它包含444个数据点； 每个数据点包含三个部分：
* ID；
* RNA序列；
* LinearFold预测结构。
