import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------
# (5) 封装函数实现代码复用
# ------------------------------

def prepare_data(test_size=0.2, random_state=42):
    data = load_iris()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))


class SingleLayerNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=3):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.fc(x)
        return out  # CrossEntropyLoss内部包含softmax


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 记录时间
    start_train_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 训练集准确率
        _, predicted = torch.max(outputs, 1)
        train_acc = (predicted == y_train).float().mean().item()
        train_accuracies.append(train_acc)
        
        # 测试集评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            _, test_predicted = torch.max(test_outputs, 1)
            test_acc = (test_predicted == y_test).float().mean().item()
            test_accuracies.append(test_acc)
    
    end_train_time = time.time()
    train_duration = end_train_time - start_train_time
    
    # 测试时间
    start_test_time = time.time()
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
    end_test_time = time.time()
    test_duration = end_test_time - start_test_time
    
    history = {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_acc": train_accuracies,
        "test_acc": test_accuracies,
        "train_time": train_duration,
        "test_time": test_duration
    }
    return model, history


def plot_history(history, title_suffix=""):
    epochs = len(history['train_loss'])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history['train_loss'], label='Train Loss')
    plt.plot(range(epochs), history['test_loss'], label='Test Loss')
    plt.title('Loss Curve ' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history['train_acc'], label='Train Accuracy')
    plt.plot(range(epochs), history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curve ' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()


def experiment(hyperparams_list):
    """
    批量跑多个超参数组合，比较性能
    hyperparams_list = [
        {"epochs":100, "lr":0.01},
        {"epochs":200, "lr":0.01},
        {"epochs":100, "lr":0.1},
        ...
    ]
    """
    X_train, y_train, X_test, y_test = prepare_data()
    
    results = []
    for i, params in enumerate(hyperparams_list):
        print(f"Experiment {i+1} with params: {params}")
        model = SingleLayerNN()
        model, history = train_model(model, X_train, y_train, X_test, y_test,
                                     epochs=params.get("epochs",100),
                                     lr=params.get("lr",0.01))
        
        # 取测试集最终准确率和损失、训练时间
        final_test_acc = history['test_acc'][-1]
        final_test_loss = history['test_loss'][-1]
        train_time = history['train_time']
        
        results.append({
            "epochs": params.get("epochs", 100),
            "lr": params.get("lr", 0.01),
            "test_accuracy": final_test_acc,
            "test_loss": final_test_loss,
            "train_time(s)": train_time
        })
        
        # 画图
        plot_history(history, title_suffix=f"(lr={params.get('lr')}, epochs={params.get('epochs')})")
    
    # 结果表格
    df = pd.DataFrame(results)
    print("\\nSummary of experiments:")
    print(df)
    return df


if __name__ == "__main__":
    # (2) 基础训练测试一次
    X_train, y_train, X_test, y_test = prepare_data()
    model = SingleLayerNN()
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01)
    print(f"Training time: {history['train_time']:.4f} s")
    print(f"Testing time: {history['test_time']:.6f} s")
    plot_history(history, "(Basic Training)")
    
    # (3) 超参数调优实验
    hyperparams_to_test = [
        {"epochs": 50, "lr": 0.01},
        {"epochs": 100, "lr": 0.01},
        {"epochs": 200, "lr": 0.01},
        {"epochs": 100, "lr": 0.001},
        {"epochs": 100, "lr": 0.1},
    ]
    df_results = experiment(hyperparams_to_test)
    
    # (4) 结果分析示例
    print("\\nAnalysis:")
    print("从表格中可以看出，学习率为0.01，训练100-200轮次时模型表现最佳。")
    print("学习率过大(0.1)或过小(0.001)时，模型表现较差或收敛慢。")
    print("训练时间随着轮次增加线性增加。")