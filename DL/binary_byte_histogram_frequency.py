import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 이미지 불러오기 함수
def load_images_from_folder(folder, img_size):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = load_img(os.path.join(folder, filename), target_size=img_size)
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)

# CNN 이진 분류기 함수
def train_binary_cnn_model(X_train, y_train, X_test, y_test, img_size, epochs=10, learning_rate=0.001):
    model = Sequential([
        Input(shape=(img_size[0], img_size[1], 3)),
        Conv2D(16, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류를 위한 출력 레이어
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype(int).flatten()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# 경로 설정
img_size = (16, 16)
normal_dir = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/Github/freqeuncy/Github"
obfuscated_normal_dirs = [
    "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/invokeCradleCrafter",
    "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/InvokeObfuscation",
    "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/IseSteroids"
]
malicious_dir = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_decoded"
obfuscated_malicious_dir = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_original"

# 첫 번째 분류기: 난독화 여부 분류기
normal_images = load_images_from_folder(normal_dir, img_size)
normal_labels = np.zeros(len(normal_images))  # 정상: 0
malicious_images = load_images_from_folder(malicious_dir, img_size)
malicious_labels = np.zeros(len(malicious_images))  # 악성: 0

obfuscated_normal_images = np.concatenate([load_images_from_folder(dir, img_size) for dir in obfuscated_normal_dirs], axis=0)
obfuscated_normal_labels = np.ones(len(obfuscated_normal_images))  # 정상 난독화: 1
obfuscated_malicious_images = load_images_from_folder(obfuscated_malicious_dir, img_size)
obfuscated_malicious_labels = np.ones(len(obfuscated_malicious_images))  # 악성 난독화: 1

X = np.concatenate((normal_images, malicious_images, obfuscated_normal_images, obfuscated_malicious_images), axis=0) / 255.0
y = np.concatenate((normal_labels, malicious_labels, obfuscated_normal_labels, obfuscated_malicious_labels), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining 1st model for Obfuscation Detection (0: 정상/악성, 1: 난독화된 데이터)")
train_binary_cnn_model(X_train, y_train, X_test, y_test, img_size)

# 두 번째 분류기: 악성 난독화 분류기
obfuscated_images = np.concatenate((obfuscated_normal_images, obfuscated_malicious_images), axis=0)
obfuscated_labels = np.concatenate((np.zeros(len(obfuscated_normal_images)), np.ones(len(obfuscated_malicious_images))), axis=0)  # 정상 난독화: 0, 악성 난독화: 1

X_train, X_test, y_train, y_test = train_test_split(obfuscated_images, obfuscated_labels, test_size=0.2, random_state=42)
print("\nTraining 2nd model for Malicious Obfuscation Detection (0: 정상 난독화, 1: 악성 난독화)")
train_binary_cnn_model(X_train, y_train, X_test, y_test, img_size)
