import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

def train_cnn_model():
    # 하이퍼파라미터 설정
    epochs = 10
    learning_rate = 0.001
    img_size = (16, 16)  # 이미지 크기

    # TensorFlow 및 GPU 확인
    print("TensorFlow version:", tf.__version__)
    print("Is GPU available:", tf.config.list_physical_devices('GPU'))

    # 이미지 경로 설정 및 라벨링
    normal_dir_1 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/Github/freqeuncy/Github"  # 정상
    obfuscated_normal_dir_1 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/invokeCradleCrafter"  # 정상 난독화
    obfuscated_normal_dir_2 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/InvokeObfuscation"  # 정상 난독화
    obfuscated_normal_dir_3 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/IseSteroids"  # 정상 난독화
    attack_dir_1 = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_decoded"  # 악성
    attack_dir_2 = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_original"  # 악성 난독화

    # 데이터 로드 및 라벨링
    normal_images_1, normal_labels_1 = load_images_from_folder(normal_dir_1, img_size, label=0)
    obfuscated_normal_images_1, obfuscated_normal_labels_1 = load_images_from_folder(obfuscated_normal_dir_1, img_size, label=1)
    obfuscated_normal_images_2, obfuscated_normal_labels_2 = load_images_from_folder(obfuscated_normal_dir_2, img_size, label=1)
    obfuscated_normal_images_3, obfuscated_normal_labels_3 = load_images_from_folder(obfuscated_normal_dir_3, img_size, label=1)
    attack_images_1, attack_labels_1 = load_images_from_folder(attack_dir_1, img_size, label=2)
    attack_images_2, attack_labels_2 = load_images_from_folder(attack_dir_2, img_size, label=3)

    # 데이터 결합
    X = np.concatenate((normal_images_1,
                        obfuscated_normal_images_1, obfuscated_normal_images_2, obfuscated_normal_images_3,
                        attack_images_1, attack_images_2), axis=0)
    y = np.concatenate((normal_labels_1,
                        obfuscated_normal_labels_1, obfuscated_normal_labels_2, obfuscated_normal_labels_3,
                        attack_labels_1, attack_labels_2), axis=0)

    # 데이터 정규화 (0 ~ 1 사이 값으로)
    X = X / 255.0

    # 학습 데이터와 테스트 데이터 분할 (80% 학습, 20% 검증)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CNN 모델 구성
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
        Dense(4, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    # 검증 세트로 성능 평가
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Confusion matrix 및 classification report 출력 (소수점 넷째 자리까지)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))


# 기본 학습 설정으로 학습 (epochs=10, learning_rate=0.001)
train_cnn_model()
