import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
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

def train_resnet50_model():
    # 하이퍼파라미터 설정
    epochs = 10
    learning_rate = 0.001
    img_size = (32, 32)  # 이미지 크기를 32x32로 설정

    # TensorFlow 및 GPU 확인
    print("TensorFlow version:", tf.__version__)
    print("Is GPU available:", tf.config.list_physical_devices('GPU'))

    # 전체 시간 측정 시작
    total_start_time = time.time()

    # 이미지 경로 설정 및 라벨링
    obfuscated_normal_dirs = [
        "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_InvokeObfuscation",
        "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_invokeCradleCrafter",
        "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_IseSteroids"
    ]
    attack_dir = "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_attack_original"

    # 데이터 로드 및 라벨링
    obfuscated_normal_images = []
    obfuscated_normal_labels = []
    for dir_path in obfuscated_normal_dirs:
        images, labels = load_images_from_folder(dir_path, img_size, label=0)
        obfuscated_normal_images.append(images)
        obfuscated_normal_labels.append(labels)
    attack_images, attack_labels = load_images_from_folder(attack_dir, img_size, label=1)

    # 데이터 결합
    X = np.concatenate((*obfuscated_normal_images, attack_images), axis=0)
    y = np.concatenate((*obfuscated_normal_labels, attack_labels), axis=0)

    # 데이터 정규화 (0 ~ 1 사이 값으로)
    X = X / 255.0

    # 학습 데이터와 테스트 데이터 분할 (80% 학습, 20% 검증)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ResNet50 모델 생성
    base_model = ResNet50(weights=None, include_top=False, input_shape=(img_size[0], img_size[1], 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 학습 시간 측정 시작
    train_start_time = time.time()

    # 모델 학습
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    # 학습 시간 측정 종료
    train_end_time = time.time()

    # 검증 세트로 성능 평가
    predictions = model.predict(X_test)
    y_pred = np.round(predictions).astype(int).reshape(-1)  # 예측값을 0 또는 1로 변환

    # Confusion matrix 및 classification report 출력 (소수점 넷째 자리까지)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # 전체 시간 측정 종료
    total_end_time = time.time()

    # 시간 출력
    print(f"Total Time: {total_end_time - total_start_time:.4f} seconds")
    print(f"Training Time: {train_end_time - train_start_time:.4f} seconds")

# 기본 학습 설정으로 학습 (epochs=10, learning_rate=0.001)
train_resnet50_model()
