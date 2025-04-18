import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import matplotlib.pyplot as plt  
import joblib

# Tải dữ liệu
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

# Kiểm tra kích thước dữ liệu
print(f"X_train shape: {X_train.shape}")  

# Xây dựng mô hình CNN với đầu vào RGB (28, 28, 3)
inputs = layers.Input(shape=(28, 28, 3))  
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(36, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Hiển thị kiến trúc mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Lưu mô hình CNN 
os.makedirs('models', exist_ok=True)
model.save('models/cnn_model.h5')

# Đánh giá mô hình CNN trước khi trích xuất đặc trưng
cnn_loss, cnn_acc = model.evaluate(X_test, y_test)
print(f"CNN Accuracy: {cnn_acc}")

# Trích xuất đặc trưng từ CNN để dùng cho KNN
feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-3].output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_features, y_train)

# Đánh giá KNN
knn_score = knn.score(X_test_features, y_test)
print(f"KNN Accuracy: {knn_score}")

# Lưu mô hình KNN
joblib.dump(knn, 'models/knn_model.pkl')

# Vẽ biểu đồ Loss và Accuracy
def plot_training_history(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Hiển thị biểu đồ
plot_training_history(history)
print("Đã huấn luyện xong mô hình!")
