import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import joblib

# Tải mô hình
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
knn_model = joblib.load('models/knn_model.pkl')
feature_extractor = tf.keras.models.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

# Hàm dự đoán
def predict_image(img_path):
    try:
        # Đọc và xử lý ảnh
        img = Image.open(img_path)
        # Kiểm tra xem mô hình dùng ảnh xám hay ảnh màu
        if cnn_model.input.shape[-1] == 1:  
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
        else:  # Ảnh màu RGB
            img = img.convert('RGB').resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28, 3) / 255.0

        # Trích xuất đặc trưng và dự đoán
        features = feature_extractor.predict(img_array, verbose=0)
        knn_pred = knn_model.predict(features)[0]
        print(f"KNN prediction: {knn_pred}")  # Debug

        # Danh sách nhãn: A-Z (0-25) và 0-9 (26-35)
        labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(10)]
        if 0 <= knn_pred < len(labels):
            return labels[knn_pred]
        else:
            return "Invalid prediction"
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return "Error"

# GUI
root = tk.Tk()
root.title("Handwritten Recognition")
root.geometry("400x400")

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        try:
            # Hiển thị ảnh
            img = Image.open(file_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            label_img.config(image=img_tk)
            label_img.image = img_tk  # Giữ tham chiếu để ảnh không bị xóa

            # Dự đoán
            pred = predict_image(file_path)
            label_pred.config(text=f"Prediction: {pred}")
        except Exception as e:
            print(f"Error in load_image: {e}")
            label_pred.config(text="Prediction: Error")

btn_load = tk.Button(root, text="Choose Image", command=load_image)
btn_load.pack(pady=10)

label_img = tk.Label(root)
label_img.pack()

label_pred = tk.Label(root, text="Prediction: None", font=("Arial", 14))
label_pred.pack(pady=10)

root.mainloop()