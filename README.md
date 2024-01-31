# README

## 运行环境

`Python`需安装前置库,见`requirements.txt`

```
torch~=1.13.1+cu116
matplotlib~=3.7.2
Pillow~=9.5.0
torchvision~=0.14.1+cu116
```

## 文件说明

文件结构如下

`data_cleansing.py`:对数据集`data`文件夹下的`{guid}.txt`里的文本内容进行数据预处理

`dict_create.py`: 对清洗后的`{guid}.txt`的文本内容进行整合,并分词创建词典.

`main.py`: 主函数, 在处理完数据后,通过该函数进行训练与预测并写入

`test.py`: 用于在`train.txt`中自行划分训练集与验证集进行模型调整测试

## 使用方法

在`Terminal`中通过Python指令直接运行

首先运行`data_cleansing.py`对数据进行处理

```
python data_cleansing.py
```

后运行`dict_create.py`生成词典`dict.txt`

```
python dict_create.py
```

最后构造完词典后运行`main.py`开始训练并对测试集进行预测

```
python main.py
```

