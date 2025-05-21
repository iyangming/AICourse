
from PIL import Image, ImageOps
import numpy as np
import os

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img)
    threshold = 128
    img = img.point(lambda p: 255 if p > threshold else 0)
    return img

def resize_to_mnist(img, size=(28, 28)):
    img = img.resize((20, 20), Image.LANCZOS)
    new_img = Image.new("L", size, 0)
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(img, upper_left)
    return new_img

def image_to_array(img):
    return np.array(img, dtype=np.uint8)

def save_array_to_file(arr, filename):
    np.savetxt(filename, arr, fmt="%d", delimiter=",")

def load_array_from_file(filename):
    return np.loadtxt(filename, delimiter=",", dtype=np.uint8)

import matplotlib.pyplot as plt

def show_images(original_img, processed_img, digit):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Original {digit}')
    axes[0].axis('off')
    
    inverted_img = ImageOps.invert(original_img)
    axes[1].imshow(inverted_img, cmap='gray')
    axes[1].set_title(f'Inverted {digit}')
    axes[1].axis('off')
    
    axes[2].imshow(processed_img, cmap='gray')
    axes[2].set_title(f'MNIST {digit}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_digits(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for digit in range(10):
        input_path = os.path.join(input_folder, f"{digit}.png")
        output_path = os.path.join(output_folder, f"{digit}.csv")
        original_img = Image.open(input_path).convert("L")
        img = preprocess_image(input_path)
        img = resize_to_mnist(img)
        arr = image_to_array(img)
        save_array_to_file(arr, output_path)
        print(f"Saved: {output_path}")
        show_images(original_img, img, digit)
