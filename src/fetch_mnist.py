import numpy as np
from sklearn.datasets import fetch_openml
import os

# Tạo thư mục trong cùng cấp với file
os.makedirs('data/raw', exist_ok=True)

# Tải dữ liệu MNIST từ OpenML
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Chuyển y sang số nguyên
y = y.astype(int)

# Lưu dữ liệu
np.save('data/raw/mnist_X.npy', X)
np.save('data/raw/mnist_y.npy', y)
print("Đã tải và lưu MNIST vào data/raw/")

