import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载并预处理数据
iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# 超参数组合
param_grid = [
    {"optimizer": "adam", "learning_rate": 0.01, "epochs": 100},
    {"optimizer": "adam", "learning_rate": 0.001, "epochs": 100},
    {"optimizer": "sgd", "learning_rate": 0.01, "epochs": 100},
    {"optimizer": "sgd", "learning_rate": 0.001, "epochs": 100},
    {"optimizer": "adam", "learning_rate": 0.001, "epochs": 200},
]

results = []

# 定义模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# 遍历每组超参数
for params in param_grid:
    model = SimpleNN(4, 3).to(device)
    criterion = nn.CrossEntropyLoss()

    if params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    elif params["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

    start_time = time.time()

    for epoch in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        loss = criterion(outputs, y_test).item()

    results.append({
        "Optimizer": params["optimizer"],
        "Learning Rate": params["learning_rate"],
        "Epochs": params["epochs"],
        "Test Accuracy": round(accuracy, 4),
        "Test Loss": round(loss, 4),
        "Training Time (s)": round(training_time, 4)
    })

# 转为 DataFrame
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="Test Accuracy", ascending=False)
print(df_sorted)

# 可视化
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.bar(range(len(df_sorted)), df_sorted["Test Accuracy"], color='skyblue')
plt.xticks(range(len(df_sorted)), [f'{opt}\nlr={lr}\nepochs={ep}' 
                                 for opt, lr, ep in zip(df_sorted["Optimizer"], df_sorted["Learning Rate"], df_sorted["Epochs"])], rotation=45, ha='right')
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison")

plt.subplot(1, 3, 2)
plt.bar(range(len(df_sorted)), df_sorted["Test Loss"], color='salmon')
plt.xticks(range(len(df_sorted)), [f'{opt}\nlr={lr}\nepochs={ep}' 
                                 for opt, lr, ep in zip(df_sorted["Optimizer"], df_sorted["Learning Rate"], df_sorted["Epochs"])], rotation=45, ha='right')
plt.ylabel("Test Loss")
plt.title("Test Loss Comparison")

plt.subplot(1, 3, 3)
plt.bar(range(len(df_sorted)), df_sorted["Training Time (s)"], color='lightgreen')
plt.xticks(range(len(df_sorted)), [f'{opt}\nlr={lr}\nepochs={ep}' 
                                 for opt, lr, ep in zip(df_sorted["Optimizer"], df_sorted["Learning Rate"], df_sorted["Epochs"])], rotation=45, ha='right')
plt.ylabel("Training Time (s)")
plt.title("Training Time Comparison")

plt.tight_layout()
plt.show()