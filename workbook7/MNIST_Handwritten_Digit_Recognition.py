# mnist_train.py
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def load_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def build_model(hidden_layers=[128, 64], optimizer='adam'):
    optimizer_dict = {
        'adam': Adam(),
        'sgd': SGD(),
        'rmsprop': RMSprop()
    }
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for units in hidden_layers:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=optimizer_dict.get(optimizer, Adam()),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(hidden_layers, optimizer, epochs=10, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_preprocess_data()
    
    model = build_model(hidden_layers, optimizer)
    
    print(f"\nTraining model with layers {hidden_layers} and optimizer '{optimizer}'")
    
    start_train = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    end_train = time.time()
    
    start_test = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    end_test = time.time()
    
    train_time = end_train - start_train
    test_time = end_test - start_test
    
    print(f"Train time: {train_time:.2f} s, Test time: {test_time:.2f} s")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return model, history, train_time, test_time, test_loss, test_acc

def plot_history(history, title_suffix=''):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy Curve {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_experiments():
    experiments = [
        {'hidden_layers': [64], 'optimizer': 'adam'},
        {'hidden_layers': [128, 64], 'optimizer': 'adam'},
        {'hidden_layers': [256, 128, 64], 'optimizer': 'adam'},
        {'hidden_layers': [128, 64], 'optimizer': 'sgd'},
        {'hidden_layers': [128, 64], 'optimizer': 'rmsprop'},
    ]
    results = []
    
    for exp in experiments:
        model, history, train_time, test_time, test_loss, test_acc = train_and_evaluate(
            exp['hidden_layers'], exp['optimizer'], epochs=10)
        
        plot_history(history, title_suffix=f"Layers:{exp['hidden_layers']} Optimizer:{exp['optimizer']}")
        
        results.append({
            'hidden_layers': exp['hidden_layers'],
            'optimizer': exp['optimizer'],
            'train_time': train_time,
            'test_time': test_time,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'model': model,
            'history': history
        })
        
    return results

def save_model(model, filename='mnist_model.h5'):
    model.save(filename)
    print(f"Model saved to {filename}")

def main():
    results = run_experiments()
    
    # 选取测试准确率最高的模型保存
    best_result = max(results, key=lambda x: x['test_accuracy'])
    
    print("\n=== Best Model Summary ===")
    print(f"Layers: {best_result['hidden_layers']}")
    print(f"Optimizer: {best_result['optimizer']}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"Test Loss: {best_result['test_loss']:.4f}")
    print(f"Training Time: {best_result['train_time']:.2f} s")
    
    save_model(best_result['model'], filename='best_mnist_model.h5')
    
    # 结果分析示例
    print("\n=== Result Analysis ===")
    print("1. 增加隐含层数和节点数通常提升准确率，但训练时间增长较快。")
    print("2. Adam优化器收敛速度快，表现稳定；SGD收敛慢但可能泛化更好。")
    print("3. 选择合适网络结构和优化器对性能影响显著。")
    print("4. 交叉熵损失与准确率曲线可帮助监控训练过程。")

if __name__ == '__main__':
    main()