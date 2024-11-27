import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# 이미지 폴더에서 PNG 파일을 읽어들여 배열로 변환하는 함수
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


def train_cnn_densenet_model():
    # 하이퍼파라미터 설정
    epochs = 10
    learning_rate = 0.001
    img_size = (32, 32)  # DenseNet121은 32x32 이미지 크기 사용

    # TensorFlow 및 GPU 확인
    print("TensorFlow version:", tf.__version__)
    print("Is GPU available:", tf.config.list_physical_devices('GPU'))

    # 이미지 경로 설정 및 라벨링
    obfuscated_normal_dirs = [
        "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_InvokeObfuscation",
        "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_invokeCradleCrafter",
        "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_IseSteroids"
    ]
    attack_dir = "D:/PowerShell/1119/byte_histogram_24.11.19/x2/frequency_attack_original"
    validation_dir = "D:/PowerShell/unknownvalidationset/rc_validation_Histogram/freqeuncy/rc_powershell"

    # 전체 시간 측정 시작
    total_start_time = time.time()

    # 데이터 로드 및 라벨링
    obfuscated_normal_images = []
    obfuscated_normal_labels = []
    for dir_path in obfuscated_normal_dirs:
        images, labels = load_images_from_folder(dir_path, img_size, label=0)
        obfuscated_normal_images.append(images)
        obfuscated_normal_labels.append(labels)

    obfuscated_normal_images = np.concatenate(obfuscated_normal_images, axis=0)
    obfuscated_normal_labels = np.concatenate(obfuscated_normal_labels, axis=0)
    attack_images, attack_labels = load_images_from_folder(attack_dir, img_size, label=1)

    # 데이터 결합
    X = np.concatenate((obfuscated_normal_images, attack_images), axis=0)
    y = np.concatenate((obfuscated_normal_labels, attack_labels), axis=0)

    # 데이터 정규화 (0 ~ 1 사이 값으로)
    X = X / 255.0

    # 학습 데이터와 테스트 데이터 분할 (80% 학습, 20% 검증)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 검증 데이터 로드 및 정규화
    validation_images, validation_labels = load_images_from_folder(validation_dir, img_size, label=1)  # 검증 데이터는 모두 label 1로 설정
    X_val = validation_images / 255.0

    # DenseNet121 모델 불러오기 (ImageNet 가중치 사용, 최상위 레이어 제외)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False  # DenseNet121 레이어 고정

    # 새로 추가한 레이어로 모델 구성
    model = Sequential([
        base_model,
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 정상 난독화와 악성 난독화 2개 라벨 분류
    ])

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
    predictions = model.predict(X_val)
    y_pred = np.round(predictions).astype(int).reshape(-1)  # 예측값을 0 또는 1로 변환

    # Confusion matrix 및 classification report 출력 (소수점 넷째 자리까지)
    y_true = validation_labels
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

    # 전체 시간 측정 종료
    total_end_time = time.time()

    # 시간 출력
    print(f"Total Time: {total_end_time - total_start_time:.4f} seconds")
    print(f"Training Time: {train_end_time - train_start_time:.4f} seconds")


# 기본 학습 설정으로 학습 (epochs=10, learning_rate=0.001)
train_cnn_densenet_model()
