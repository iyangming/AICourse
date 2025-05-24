import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import random

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat

def build_optimized_model():
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0005),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_random_images(model, x_test, y_test):
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    plt.figure(figsize=(12,6))
    for i in range(10):
        idx = random.randint(0, len(x_test) - 1)
        img = x_test[idx]
        true_label = y_test[idx][0]
        pred = model.predict(img.reshape(1,32,32,3))
        pred_label = np.argmax(pred)
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"真: {class_names[true_label]}\n预测: {class_names[pred_label]}")
    plt.tight_layout()
    plt.show()

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # For MacOS
plt.rcParams['axes.unicode_minus'] = False


def main():
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_and_preprocess_data()
    
    model = build_optimized_model()

    start_train = time.time()
    history = model.fit(x_train, y_train_cat, epochs=15, batch_size=64, validation_split=0.2, verbose=2)
    end_train = time.time()

    start_test = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
    end_test = time.time()

    print(f"\n训练时间: {end_train - start_train:.2f} 秒")
    print(f"测试时间: {end_test - start_test:.2f} 秒")
    print(f"测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")

    plot_history(history)

    model.save('Cifar10.h5')
    print("模型已保存为 Cifar10.h5")

    print("\n加载模型并预测 10 张随机图片：")
    loaded_model = load_model('Cifar10.h5')
    predict_random_images(loaded_model, x_test, y_test)

if __name__ == '__main__':
    main()