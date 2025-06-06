# 华为MindSpore课程章节测验题（共100题）

---

## 第1章：MindSpore概述与安装配置（共20题）

第1章：MindSpore概览与环境配置（共10题）

	1.	MindSpore 最初由哪家公司发布？
	•	A. 百度
	•	B. 腾讯
	•	C. 华为 ✅
	•	D. 阿里巴巴
	2.	MindSpore 的主要执行模式为：
	•	A. 静态图
	•	B. 动态图 ✅
	•	C. 混合图
	•	D. 延迟执行
	3.	以下哪个不是 MindSpore 框架支持的设备平台？
	•	A. Ascend
	•	B. GPU
	•	C. CPU
	•	D. TPU ✅
	4.	使用 MindSpore 安装包的推荐 Python 版本是：
	•	A. Python 2.7
	•	B. Python 3.6 ✅
	•	C. Python 3.5
	•	D. Python 3.8
	5.	MindSpore 支持的分布式通信库是：
	•	A. NCCL
	•	B. HCCL ✅
	•	C. DDP
	•	D. RPC
	6.	MindSpore 模型训练推荐使用哪种文件格式保存模型参数？
	•	A. .ckpt ✅
	•	B. .h5
	•	C. .pth
	•	D. .air
	7.	MindSpore 的官网提供的 Notebook 教学平台是：
	•	A. Jupiter
	•	B. ModelArts ✅
	•	C. AI Studio
	•	D. Google Colab
	8.	在 MindSpore 中，执行图编译和优化的模块是：
	•	A. Graph Executor
	•	B. ME模块 ✅
	•	C. Frontend Parser
	•	D. Session Manager
	9.	MindSpore 框架最底层的执行后端为：
	•	A. TensorRT
	•	B. GE ✅
	•	C. TF Lite
	•	D. TVM
	10.	MindSpore 安装命令中哪个是正确的？
	•	A. pip install mindspore ✅
	•	B. conda install mindspore
	•	C. apt-get install mindspore
	•	D. pip mindspore install



## 第2章：数据处理与数据集构建（共20题）


第2章：Tensor基础与API使用（共20题）
	

	11.	MindSpore 中，创建一个全零Tensor使用的函数是：
	•	A. mindspore.zeros() ✅
	•	B. mindspore.ones()
	•	C. mindspore.empty()
	•	D. mindspore.full()
	
	12.	以下哪个操作用于改变Tensor的形状？
	
	•	A. reshape() ✅
	•	B. transpose()
	•	C. expand_dims()
	•	D. squeeze()
	
	13.	使用 MindSpore 的 API，计算两个Tensor元素乘积的函数是：
	
	•	A. mindspore.matmul()
	•	B. mindspore.mul() ✅
	•	C. mindspore.add()
	•	D. mindspore.dot()
	
	14.	Tensor的默认数据类型通常是：
	
	•	A. int32
	•	B. float32 ✅
	•	C. float64
	•	D. int64
	
	15.	MindSpore 中，获取 Tensor 形状信息的属性是：
	
	•	A. .size()
	•	B. .shape ✅
	•	C. .dim()
	•	D. .length()
	
	16.	若要在 Tensor 维度 0 上拼接两个 Tensor，使用的函数是：
	
	•	A. mindspore.cat(tensors, dim=0)
	•	B. mindspore.concat(tensors, axis=0) ✅
	•	C. mindspore.stack(tensors, dim=0)
	•	D. mindspore.join(tensors, dim=0)
	
	17.	以下哪个方法可以把 Tensor 转换成 NumPy 数组？
	
	•	A. .asnumpy() ✅
	•	B. .numpy()
	•	C. .tolist()
	•	D. .toarray()
	
	18.	Tensor中，计算所有元素和的函数是：
	
	•	A. sum() ✅
	•	B. mean()
	•	C. prod()
	•	D. max()
	
	19.	在MindSpore中，随机生成一个服从正态分布的Tensor函数是：
	
	•	A. mindspore.randn() ✅
	•	B. mindspore.random()
	•	C. mindspore.uniform()
	•	D. mindspore.normal()
	
	20.	MindSpore 中，哪个API用于计算两个 Tensor 的矩阵乘法？
	
	•	A. matmul() ✅
	•	B. mul()
	•	C. dot()
	•	D. mm()
	
	21.	下面哪个操作可以沿着指定轴对 Tensor 求最大值？
	
	•	A. mindspore.max(tensor, axis=1) ✅
	•	B. mindspore.argmax(tensor)
	•	C. mindspore.maximum(tensor)
	•	D. mindspore.reduce_max(tensor)
	
	22.	MindSpore中用于计算Tensor均值的函数是：
	
	•	A. mean() ✅
	•	B. average()
	•	C. median()
	•	D. mode()
	
	23.	下面哪个操作可以用来给Tensor增加一个维度？
	
	•	A. unsqueeze() ✅
	•	B. squeeze()
	•	C. flatten()
	•	D. reshape()
	
	24.	使用 MindSpore API，如何将Tensor从CPU传到Ascend设备？
	
	•	A. .to("Ascend") ✅
	•	B. .device("Ascend")
	•	C. .copy_to("Ascend")
	•	D. .move("Ascend")
	
	25.	下列哪个函数可以计算两个Tensor的逐元素最大值？
	
	•	A. maximum() ✅
	•	B. max()
	•	C. clip()
	•	D. max_pool()
	
	26.	MindSpore中的Tensor支持广播机制，广播是指：
	
	•	A. 自动调整Tensor维度使其兼容运算 ✅
	•	B. 自动复制Tensor
	•	C. 随机扩展Tensor形状
	•	D. 执行分布式训练
	
	27.	Tensor中的.item()方法用于：
	
	•	A. 转换成标量 ✅
	•	B. 返回Tensor大小
	•	C. 复制Tensor数据
	•	D. 查看元素索引
	
	28.	MindSpore中，执行Tensor按行拼接使用的函数是：
	
	•	A. concat(axis=1) ✅
	•	B. stack(axis=1)
	•	C. join(axis=1)
	•	D. append(axis=1)
	
	29.	Tensor中用来计算元素平方根的函数是：
	
	•	A. sqrt() ✅
	•	B. pow(0.5)
	•	C. rsqrt()
	•	D. square()
	
	30.	MindSpore中，获取Tensor数据类型的属性是：
	
	•	A. .dtype ✅
	•	B. .type
	•	C. .data_type
	•	D. .kind
	
	31.	MindSpore API中，执行Tensor转置的函数是：
	
	•	A. transpose() ✅
	•	B. permute()
	•	C. flip()
	•	D. reverse()
	
	32.	创建一个随机均匀分布的Tensor使用的函数是：
	
	•	A. mindspore.uniform() ✅
	•	B. mindspore.randn()
	•	C. mindspore.random()
	•	D. mindspore.rand()
	
	33.	Tensor的 .copy() 方法作用是：
	
	•	A. 复制Tensor数据 ✅
	•	B. 复制引用
	•	C. 清空Tensor
	•	D. 转换为NumPy
	
	34.	在MindSpore中，以下哪个是创建1维Tensor的正确代码？
	
	•	A. Tensor([1,2,3]) ✅
	•	B. Tensor([[1,2,3]])
	•	C. Tensor([ [1],[2],[3] ])
	•	D. Tensor((1,2,3))
	
	35.	MindSpore中，Tensor的维度个数称为：
	
	•	A. size
	•	B. rank ✅
	•	C. length
	•	D. shape
	
	36.	MindSpore中的 .expand_dims() 函数用于：
	
	•	A. 增加Tensor的维度 ✅
	•	B. 减少Tensor的维度
	•	C. 转置Tensor
	•	D. 变换数据类型
	
	37.	下面哪个函数可以把Tensor展平为一维？
	
	•	A. flatten() ✅
	•	B. reshape()
	•	C. squeeze()
	•	D. expand_dims()
	
	38.	MindSpore中，使用 .detach() 方法的作用是：
	
	•	A. 返回一个不计算梯度的新Tensor ✅
	•	B. 复制Tensor
	•	C. 转换为NumPy
	•	D. 计算Tensor梯度
	
	39.	MindSpore中，以下哪个操作可以将Tensor元素限制在一个区间内？
	
	•	A. clip_by_value() ✅
	•	B. clamp()
	•	C. bound()
	•	D. restrict()
	
	40.	MindSpore中，计算Tensor中非零元素个数的函数是：
	
	•	A. count_nonzero() ✅
	•	B. nonzero()
	•	C. sum()
	•	D. where()

⸻

这20题你先保存，接下来我会继续发第3章的20题！
需要调整格式或加注释提醒，随时说。

## 第3章：构建与训练神经网络（共20题）
第3章：自动求导机制与计算图（共20题）
	

	41.	MindSpore中，自动求导的核心模块是：
	•	A. ms.ops.grad()
	•	B. mindspore.grad() ✅
	•	C. autograd()
	•	D. GradientTape()
	
	42.	在MindSpore中，使用value_and_grad函数的主要作用是：
	
	•	A. 获取参数值
	•	B. 获取Tensor的数值
	•	C. 同时返回函数值和梯度 ✅
	•	D. 返回参数类型
	
	43.	以下哪种计算是MindSpore自动求导不支持的？
	
	•	A. 标量函数
	•	B. 张量函数
	•	C. 控制流中包含梯度断点 ✅
	•	D. 多输入多输出函数
	
	44.	自动求导中，若不希望某层参与梯度计算应使用：
	
	•	A. .detach() ✅
	•	B. .requires_grad=False
	•	C. .disable_grad()
	•	D. .clear()
	
	45.	MindSpore中的反向传播通常在什么阶段执行？
	
	•	A. 模型定义
	•	B. 数据预处理
	•	C. 训练阶段 ✅
	•	D. 推理阶段
	
	46.	在使用mindspore.grad()时，常见的第一个参数是：
	
	•	A. Tensor
	•	B. 参数名称
	•	C. 被求导的函数 ✅
	•	D. 数据路径
	
	47.	若想冻结模型参数防止更新，应设置哪些参数属性？
	
	•	A. requires_grad = False ✅
	•	B. is_train = False
	•	C. trainable = False
	•	D. no_grad = True
	
	48.	下列哪个方法可以实现多输入函数的梯度计算？
	
	•	A. grad(fn, (inputs1, inputs2)) ✅
	•	B. grad(fn, *inputs)
	•	C. grad(fn).backward()
	•	D. grad.backward(inputs)
	
	49.	MindSpore中，动态图和静态图的最大区别是：
	
	•	A. 前者性能高
	•	B. 后者只用于推理
	•	C. 动态图按需构建 ✅
	•	D. 静态图不能自动求导
	
	50.	MindSpore 中的 GradOperation 作用是：
	
	•	A. 自动微分构造器 ✅
	•	B. 构建模型参数
	•	C. 定义计算图
	•	D. 模型初始化操作
	
	51.	计算图中的“节点”通常对应：
	
	•	A. 张量数据
	•	B. 运算操作 ✅
	•	C. 梯度值
	•	D. 损失函数
	
	52.	在构建计算图时，下面哪个操作会成为一个“边”？
	
	•	A. 模型权重
	•	B. 前向传递 ✅
	•	C. 打印日志
	•	D. 存储变量
	
	53.	MindSpore中，反向传播的起点通常是：
	
	•	A. 输入层
	•	B. 输出Tensor ✅
	•	C. 参数列表
	•	D. 中间结果
	
	54.	在静态图模式下，MindSpore的计算图在什么时间被构建？
	
	•	A. 运行时
	•	B. 编译时 ✅
	•	C. 推理时
	•	D. 加载模型时
	
	55.	with ms.GradientTape(): 是哪个框架的用法？
	
	•	A. TensorFlow ✅
	•	B. MindSpore
	•	C. PyTorch
	•	D. JAX
	
	56.	mindspore.ops.grad 默认返回什么？
	
	•	A. 损失值
	•	B. 权重
	•	C. 梯度函数 ✅
	•	D. 更新后的模型
	
	57.	MindSpore中的 set_grad() 的作用是：
	
	•	A. 打开梯度计算 ✅
	•	B. 关闭图优化
	•	C. 设置Tensor维度
	•	D. 调整图结构
	
	58.	在求导中常见的链式法则在MindSpore中是如何体现的？
	
	•	A. 构建多个计算图
	•	B. 梯度函数自动组合 ✅
	•	C. 手动定义梯度
	•	D. 使用低级接口
	
	59.	计算图在优化后可以减少什么？
	
	•	A. 运算次数 ✅
	•	B. 参数数量
	•	C. 网络深度
	•	D. 模型大小
	
	60.	以下哪个功能可以加速MindSpore计算图执行效率？
	
	•	A. JIT编译 ✅
	•	B. 打开日志
	•	C. 多次构建图
	•	D. 使用循环结构

⸻

✅：表示正确答案
如需继续第4章，请告诉我，内容包含【模型构建与训练流程】的测试题（61～80）。也可以随时打包生成下载链接。

## 第4章：模型评估与调优（共20题）
第4章：模型构建与训练流程（共20题）
	

	61.	在MindSpore中定义神经网络模型，推荐继承哪个基类？
	•	A. ms.Graph
	•	B. ms.Model
	•	C. nn.Cell ✅
	•	D. ModelBase
	
	62.	构建模型时，construct()方法的主要作用是：
	
	•	A. 定义超参数
	•	B. 实现前向计算逻辑 ✅
	•	C. 保存网络权重
	•	D. 进行梯度更新
	
	63.	模型训练的基本流程不包括以下哪项？
	
	•	A. 定义网络结构
	•	B. 定义损失函数
	•	C. 编写推理函数 ✅
	•	D. 选择优化器
	
	64.	MindSpore的模型训练类是：
	
	•	A. TrainLoop
	•	B. Trainer
	•	C. Model ✅
	•	D. Session
	
	65.	以下哪个模块用于构建全连接层？
	
	•	A. nn.Dense ✅
	•	B. nn.Conv2d
	•	C. nn.Flatten
	•	D. nn.LSTM
	
	66.	以下哪个模块适用于多分类交叉熵损失？
	
	•	A. nn.L1Loss()
	•	B. nn.MSELoss()
	•	C. nn.SoftmaxCrossEntropyWithLogits() ✅
	•	D. nn.CrossEntropyWithSigmoid()
	
	67.	为了提高训练性能，MindSpore建议使用的数据格式是：
	
	•	A. CSV
	•	B. JSON
	•	C. MindRecord ✅
	•	D. Numpy
	
	68.	在训练模型前，通常使用model.train()时必须传入：
	
	•	A. 测试数据集
	•	B. 损失函数
	•	C. 训练轮数 ✅
	•	D. 参数字典
	
	69.	在模型训练中，如果不指定优化器，可能导致：
	
	•	A. 自动推理失败
	•	B. 权重初始化失败
	•	C. 无法更新参数 ✅
	•	D. 模型精度过高
	
	70.	以下哪项不是Model.train()的常见参数？
	
	•	A. epoch
	•	B. train_dataset
	•	C. callbacks
	•	D. optimizer ✅
	
	71.	训练中用来监控精度的组件通常是：
	
	•	A. LossMonitor
	•	B. Accuracy ✅
	•	C. TimeMonitor
	•	D. DatasetSinkMode
	
	72.	MindSpore中，优化器的主要作用是：
	
	•	A. 减少计算图
	•	B. 初始化网络
	•	C. 更新模型参数 ✅
	•	D. 降低数据噪声
	
	73.	为了避免梯度爆炸，可以使用什么技术？
	
	•	A. 权重正则化
	•	B. 梯度剪裁 ✅
	•	C. 数据增强
	•	D. 学习率预热
	
	74.	模型训练完成后，进行模型评估的函数是：
	
	•	A. evaluate() ✅
	•	B. test()
	•	C. infer()
	•	D. predict()
	
	75.	训练过程中的callback函数用于：
	
	•	A. 调整网络层数
	•	B. 自动记录和监控过程信息 ✅
	•	C. 编译模型
	•	D. 管理梯度
	
	76.	常用于保存训练中间模型的是：
	
	•	A. LossMonitor
	•	B. CheckpointConfig ✅
	•	C. DataMonitor
	•	D. DatasetHelper
	
	77.	以下哪个指标更适合分类问题的评估？
	
	•	A. 均方误差
	•	B. 准确率 ✅
	•	C. 均方根
	•	D. 平均绝对误差
	
	78.	若使用model.eval()方法，其返回值通常为：
	
	•	A. 权重矩阵
	•	B. 模型结构
	•	C. 性能指标 ✅
	•	D. 优化器
	
	79.	在MindSpore中设置训练日志间隔的参数通常位于：
	
	•	A. TrainConfig
	•	B. Callback ✅
	•	C. Dataset
	•	D. Optimizer
	
	80.	以下哪个组件可用于动态调整学习率？
	
	•	A. FixedLoss
	•	B. CallbackMonitor
	•	C. LearningRateScheduler ✅
	•	D. TrainOneStepCell



## 第5章：模型保存与推理部署（共20题）




第5章：模型保存与推理部署（共20题）
	

	81.	在MindSpore中保存模型参数使用的API是：
	•	A. save_model()
	•	B. save_checkpoint() ✅
	•	C. save_weights()
	•	D. export_model()
	
	82.	以下哪个文件格式是MindSpore保存模型权重常用的格式？
	
	•	A. .pth
	•	B. .pb
	•	C. .ckpt ✅
	•	D. .json
	
	83.	要加载已训练模型参数，需使用哪个函数？
	
	•	A. model.load()
	•	B. load_ckpt()
	•	C. load_checkpoint() ✅
	•	D. restore_model()
	
	84.	推理阶段通常不需要以下哪一部分？
	
	•	A. 模型结构
	•	B. 训练数据 ✅
	•	C. 模型权重
	•	D. 推理输入
	
	85.	在保存模型的同时保存网络结构，应该使用：
	
	•	A. save_weights()
	•	B. export() ✅
	•	C. load_checkpoint()
	•	D. construct_model()
	
	86.	MindSpore中用于模型导出的格式不包括：
	
	•	A. AIR
	•	B. ONNX
	•	C. TFLite ✅
	•	D. MINDIR
	
	87.	推理部署中，使用.air格式的模型主要针对以下平台：
	
	•	A. Windows PC
	•	B. Ascend AI处理器 ✅
	•	C. Android
	•	D. iOS
	
	88.	export() 函数的第三个参数是指定：
	
	•	A. 导出路径
	•	B. 模型名称
	•	C. 输出格式 ✅
	•	D. 权重参数
	
	89.	导出模型文件前必须先：
	
	•	A. 关闭梯度计算
	•	B. 设置模型为评估模式 ✅
	•	C. 编译模型
	•	D. 构建数据集
	
	90.	模型推理一般通过哪个函数执行？
	
	•	A. model.infer()
	•	B. model.eval()
	•	C. model.predict() ✅
	•	D. model.run()
	
	91.	在部署模型前，对输入数据的预处理操作不包括：
	
	•	A. 归一化
	•	B. 张量转换
	•	C. 损失函数设置 ✅
	•	D. 尺寸调整
	
	92.	MindSpore支持的离线推理工具是：
	
	•	A. MindRT
	•	B. Ascend-Predict
	•	C. MindSpore Lite ✅
	•	D. MSRun
	
	93.	load_param_into_net()函数的作用是：
	
	•	A. 定义网络结构
	•	B. 创建新模型
	•	C. 加载权重到网络 ✅
	•	D. 保存参数为.ckpt文件
	
	94.	MindSpore推理部署中，常用于边缘设备的推理引擎是：
	
	•	A. TensorRT
	•	B. MindIR Lite
	•	C. MindSpore Lite ✅
	•	D. CoreML
	
	95.	推理阶段的主要目标是：
	
	•	A. 调整学习率
	•	B. 训练网络
	•	C. 输出预测结果 ✅
	•	D. 计算损失函数
	
	96.	以下哪个模块用于定义推理阶段的输入输出张量？
	
	•	A. mindspore.Tensor ✅
	•	B. mindspore.Dataset
	•	C. mindspore.export
	•	D. mindspore.DataLoader
	
	97.	推理部署中模型过大，最常用的优化技术是：
	
	•	A. 数据增强
	•	B. 网络剪枝 ✅
	•	C. 模型评估
	•	D. 可视化
	
	98.	MINDIR格式模型在导出时最适合哪类应用？
	
	•	A. 在线服务部署 ✅
	•	B. 图像采集
	•	C. 音频生成
	•	D. 实时训练
	
	99.	若推理性能不佳，应优先考虑：
	
	•	A. 更换数据源
	•	B. 减少批大小 ✅
	•	C. 增加损失函数复杂度
	•	D. 更换操作系统
	
	100.	MindSpore模型部署的完整流程不包括以下哪项？
	
	•	A. 构建数据集 ✅
	•	B. 导出模型
	•	C. 加载模型
	•	D. 运行预测

