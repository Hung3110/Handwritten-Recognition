import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Đọc dữ liệu A-Z
az_data = pd.read_csv('data/raw/A_Z Handwritten Data.csv')
az_labels = az_data.iloc[:, 0].values  # Nhãn từ 0-25 (A-Z)
az_images = az_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # Chuẩn hóa

# Đọc dữ liệu 0-9
mnist_X = np.load('data/raw/mnist_X.npy')
mnist_y = np.load('data/raw/mnist_y.npy', allow_pickle=True)  
mnist_images = mnist_X.reshape(-1, 28, 28, 1) / 255.0
mnist_labels = mnist_y.astype(int) + 26  # Dịch nhãn thành 26-35

# Chuyển ảnh xám sang RGB giả lập (sao chép kênh xám thành 3 kênh)
az_images_rgb = np.repeat(az_images, 3, axis=-1)  
mnist_images_rgb = np.repeat(mnist_images, 3, axis=-1)  

# Gộp dữ liệu
X = np.concatenate([az_images_rgb, mnist_images_rgb], axis=0)
y = np.concatenate([az_labels, mnist_labels], axis=0)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augmentation 
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Lưu dữ liệu đã xử lý
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)
print("Đã chuẩn bị dữ liệu xong!")