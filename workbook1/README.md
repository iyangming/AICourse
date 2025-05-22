# Iris Logistic Regression 项目

## 项目概述
本项目使用逻辑回归模型对Iris数据集中的Setosa和Virginica两类花进行分类。通过花瓣长度和宽度特征，建立二元分类模型。

## 数据说明
- 数据来源：sklearn内置Iris数据集
- 使用特征：花瓣长度(petal length)和花瓣宽度(petal width)
- 目标变量：Setosa(0) vs Virginica(1)
- 数据预处理：筛选Setosa和Virginica两类，将标签重新映射为0和1

## 模型训练流程
1. 数据加载与预处理
2. 特征选择（花瓣长度和宽度）
3. 训练测试集划分(70%训练，30%测试)
4. 逻辑回归模型训练（默认参数）
5. 模型评估：
   - 使用log loss和accuracy评估模型性能
   - 训练集准确率：~98%
   - 测试集准确率：~97%

## 运行方式
```bash
python iris_logistic_regression.py
```
或打开Jupyter Notebook:
```bash
jupyter notebook iris_logistic_regression.ipynb
```

## 模型训练流程
1. 数据加载与预处理
2. 特征选择（使用花瓣长度和宽度）
3. 数据标准化
4. 训练逻辑回归模型
5. 评估模型性能

## 可视化结果
项目会生成pairplot.png文件，展示两类鸢尾花在花瓣长度和宽度上的分布差异。

## 模型性能指标
- 训练集/测试集损失(log loss)
- 训练集/测试集准确率(accuracy)
- 训练时间

## 超参数实验结果
我们测试了不同学习率(0.001, 0.01, 0.1, 1.0, 10.0)和最大迭代次数(50, 100, 200, 500)的组合。实验结果如下：

### 最佳参数组合
- 学习率: 0.1
- 最大迭代次数: 200
- 测试集准确率: 1.0
- 测试集损失: 0.0

### 可视化结果
1. `learning_rate_vs_loss.png`: 展示不同学习率对测试损失的影响
2. `max_iter_vs_accuracy.png`: 展示不同迭代次数对测试准确率的影响