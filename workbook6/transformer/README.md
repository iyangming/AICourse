# Transformer项目

本项目基于MindSpore框架实现Transformer模型，用于机器翻译任务。

## 环境要求
- Python 3.7+
- MindSpore 2.3.1

## 数据集
使用Multi30K数据集，包含德语到英语的翻译对。

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练：`python train.py`
3. 运行推理：`python inference.py`

## 项目结构
```
transformer/
├── datasets/          # 数据集
├── transformer-new.ipynb  # 主要代码
└── README.md         # 项目说明
```

## 参考
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [MindSpore官方文档](https://www.mindspore.cn/docs/zh-CN/r2.3/index.html)