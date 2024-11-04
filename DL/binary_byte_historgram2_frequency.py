import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 이미지 폴더에서 PNG 파일을 읽어 배열로 변환하는 함수
def load_images_from_folder(folder, img_size, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = load_img(os.path.join(folder, filename), target_size=img_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

def first_stage_classifier(X, y, img_size):
    # 첫 번째 분류기: 정상과 악성 (0과 2) vs. 정상 난독화와 악성 난독화 (1과 3)
    y_binary = np.where((y == 0) | (y == 2), 0, 1)  # 0, 2 -> 0; 1, 3 -> 1

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(img_size[0], img_size[1], 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    y_pred = np.argmax(model.predict(X), axis=1)

    print("First Stage - Confusion Matrix:\n", confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1)))
    print("\nFirst Stage - Classification Report:\n", classification_report(y_test, np.argmax(model.predict(X_test), axis=1), digits=4))

    # 난독화로 분류된 데이터만 필터링
    obfuscated_indices = np.where(y_pred == 1)[0]
    X_obfuscated = X[obfuscated_indices]
    y_obfuscated = y[obfuscated_indices]  # y도 동일한 인덱스로 필터링

    return X_obfuscated, y_obfuscated


def second_stage_classifier(X, y, img_size):
    # 두 번째 분류기: 정상 난독화 vs. 악성 난독화 (1 vs. 3)
    y_binary = np.where(y == 1, 0, 1)  # 1 -> 0; 3 -> 1

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(img_size[0], img_size[1], 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    print("Second Stage - Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nSecond Stage - Classification Report:\n", classification_report(y_test, y_pred, digits=4))

def main():
    img_size = (16, 16)
    # 경로 설정
    normal_dir = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/Github/freqeuncy/Github"
    obfuscated_normal_dirs = [
        "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/invokeCradleCrafter",
        "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/InvokeObfuscation",
        "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/IseSteroids"
    ]
    malicious_dir = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_decoded"
    obfuscated_malicious_dir = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_original"

    # 데이터 로드 및 라벨링
    normal_images, normal_labels = load_images_from_folder(normal_dir, img_size, label=0)
    obfuscated_normal_images_1, obfuscated_normal_labels_1 = load_images_from_folder(obfuscated_normal_dirs[0], img_size, label=1)
    obfuscated_normal_images_2, obfuscated_normal_labels_2 = load_images_from_folder(obfuscated_normal_dirs[1], img_size, label=1)
    obfuscated_normal_images_3, obfuscated_normal_labels_3 = load_images_from_folder(obfuscated_normal_dirs[2], img_size, label=1)
    malicious_images, malicious_labels = load_images_from_folder(malicious_dir, img_size, label=2)
    obfuscated_malicious_images, obfuscated_malicious_labels = load_images_from_folder(obfuscated_malicious_dir, img_size, label=3)

    # 데이터 결합
    X = np.concatenate((normal_images, obfuscated_normal_images_1, obfuscated_normal_images_2, obfuscated_normal_images_3, malicious_images, obfuscated_malicious_images), axis=0)
    y = np.concatenate((normal_labels, obfuscated_normal_labels_1, obfuscated_normal_labels_2, obfuscated_normal_labels_3, malicious_labels, obfuscated_malicious_labels), axis=0)

    # 첫 번째 분류기 실행
    print("Running First Stage Classifier...")
    X_obfuscated, y_obfuscated = first_stage_classifier(X, y, img_size)

    # 두 번째 분류기 실행
    print("Running Second Stage Classifier...")
    second_stage_classifier(X_obfuscated, y_obfuscated, img_size)

main()
