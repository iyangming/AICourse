import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import time

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return df

def visualize_data(df):
    binary_df = df[df['species'].isin(['setosa', 'virginica'])]
    sns.pairplot(binary_df, hue='species', diag_kind='hist')
    plt.suptitle("Setosa vs Virginica", y=1.02)
    plt.tight_layout()
    plt.savefig("pairplot.png")
    plt.close()

def train_model(df, learning_rate=None, max_iter=100):
    binary_df = df[df['species'].isin(['setosa', 'virginica'])]
    X = binary_df[['petal length (cm)', 'petal width (cm)']].values
    y = binary_df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver='saga', max_iter=max_iter, C=1/learning_rate if learning_rate else 1.0)
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()

    y_train_pred = model.predict_proba(X_train_scaled)
    y_test_pred = model.predict_proba(X_test_scaled)
    train_loss = log_loss(y_train, y_train_pred)
    test_loss = log_loss(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, np.argmax(y_train_pred, axis=1))
    test_acc = accuracy_score(y_test, np.argmax(y_test_pred, axis=1))
    training_time = end_time - start_time

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "training_time": training_time
    }

def run_hyperparameter_experiment():
    df = load_data()
    visualize_data(df)
    
    learning_rates = [0.01, 0.1, 1.0]
    max_iters = [200, 500, 1000]
    
    results = []
    for lr in learning_rates:
        for iters in max_iters:
            result = train_model(df, learning_rate=lr, max_iter=iters)
            result['learning_rate'] = lr
            result['max_iter'] = iters
            results.append(result)
    
    # 转换为DataFrame便于分析
    df_results = pd.DataFrame(results)
    
    # 绘制学习率对损失的影响
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_results, x='learning_rate', y='test_loss', hue='max_iter', marker='o')
    plt.xscale('log')
    plt.title('Test Loss vs Learning Rate (by Max Iterations)')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Test Loss')
    plt.savefig('learning_rate_vs_loss.png')
    plt.close()
    
    # 绘制迭代次数对准确率的影响
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_results, x='max_iter', y='test_acc', hue='learning_rate', marker='o')
    plt.title('Test Accuracy vs Max Iterations (by Learning Rate)')
    plt.xlabel('Max Iterations')
    plt.ylabel('Test Accuracy')
    plt.savefig('max_iter_vs_accuracy.png')
    plt.close()
    
    # 找出最佳参数组合
    best_idx = df_results['test_acc'].idxmax()
    best_params = df_results.loc[best_idx]
    
    print("=== 最佳参数组合 ===")
    print(f"学习率: {best_params['learning_rate']:.4f}")
    print(f"最大迭代次数: {best_params['max_iter']}")
    print(f"测试集准确率: {best_params['test_acc']:.4f}")
    print(f"测试集损失: {best_params['test_loss']:.4f}")
    print(f"训练集损失: {best_params['train_loss']:.4f}")
    print(f"训练时间: {best_params['training_time']:.4f}秒")
    
    # 保存所有结果到CSV文件
    df_results.to_csv('hyperparameter_results.csv', index=False)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 6))
    
    # 获取训练数据
    binary_df = df[df['species'].isin(['setosa', 'virginica'])]
    X = binary_df[['petal length (cm)', 'petal width (cm)']].values
    y = binary_df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 训练最佳模型
    best_model = LogisticRegression(solver='saga', 
                                  max_iter=int(best_params['max_iter']), 
                                  C=1/best_params['learning_rate'])
    best_model.fit(X_train_scaled, y_train)
    
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    Z = best_model.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和散点图
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu_r', edgecolor='white')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Decision Boundary of Logistic Regression')
    plt.colorbar(label='Probability of Virginica')
    plt.savefig('decision_boundary.png')
    plt.close()
    
    return df_results

if __name__ == "__main__":
    experiment_results = run_hyperparameter_experiment()