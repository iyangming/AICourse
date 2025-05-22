# Iris Softmax分类项目

## 项目描述
本项目演示了使用Softmax回归(通过LogisticRegression实现)对Iris数据集进行多分类。模型基于花瓣测量数据预测鸢尾花种类。

## 数据集
- 数据源: sklearn.datasets.load_iris()
- 使用特征: 花瓣长度和宽度(2个特征)
- 目标: 鸢尾花种类(3个类别)

## 模型细节
- 算法: 使用多项式(softmax)损失的LogisticRegression
- 求解器: lbfgs
- 默认超参数: C=100, max_iter=1000

## 主要发现
1. **测试准确率**: 使用默认参数达到约0.97的准确率
2. **特征重要性**: 仅花瓣测量数据即可提供优秀的分类效果
3. **超参数比较**: 评估了学习率[0.01, 0.1, 1]和最大迭代次数[50, 100, 200]

## 使用方法
1. 运行脚本:
```bash
python iris_softmax_classification.py
```
2. 脚本将:
   - 训练模型
   - 打印测试准确率
   - 显示分类边界可视化
   - 比较不同特征组合
   - 评估训练集比例

## 可视化
脚本生成一个图表，显示基于花瓣测量数据的三种鸢尾花之间的决策边界。

## 运行要求
- Python 3.x
- numpy
- matplotlib
- scikit-learn
- pandas