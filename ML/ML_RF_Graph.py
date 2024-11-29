import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

# AST 파일을 읽어 리스트로 반환하는 함수
def read_ast_files(directory):
    ast_texts = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            ast_texts.append(file.read())
    return ast_texts

# 디렉토리에서 데이터를 특정 라벨로 모아주는 함수
def gather_data_from_directories(directories, label):
    all_texts = []
    all_labels = []
    for directory in directories:
        ast_texts = read_ast_files(directory)
        all_texts.extend(ast_texts)
        all_labels.extend([label] * len(ast_texts))
    return all_texts, all_labels

# 데이터 경로 설정
benign_ast_dirs = [
    'D:/PowerShell/Extraction/GithubGist/GithubGist(ast)',
    'D:/PowerShell/Extraction/Technet/Technet(ast)',
    'D:/PowerShell/Extraction/PoshCode/PoshCode(ast)'
]
obfuscated_benign_ast_dirs = [
    'D:/PowerShell/Extraction/InvokeCradleCrafter/InvokeCradleCrafter(ast)',
    'D:/PowerShell/Extraction/InvokeObfuscation/InvokeObfuscation(ast)',
    'D:/PowerShell/Extraction/IseSteroids/IseSteroids(ast)'
]
malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/decoded/decoded(ast)'
obfuscated_malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/encoded/mentor(ast)'

# 데이터 로드
benign_texts, benign_labels = gather_data_from_directories(benign_ast_dirs, label=0)
obfuscated_benign_texts, obfuscated_benign_labels = gather_data_from_directories(obfuscated_benign_ast_dirs, label=1)
malicious_texts, malicious_labels = gather_data_from_directories([malicious_ast_dir], label=2)
obfuscated_malicious_texts, obfuscated_malicious_labels = gather_data_from_directories([obfuscated_malicious_ast_dir], label=3)
all_texts = benign_texts + obfuscated_benign_texts + malicious_texts + obfuscated_malicious_texts
all_labels = benign_labels + obfuscated_benign_labels + malicious_labels + obfuscated_malicious_labels
y = np.array(all_labels)

# AST 3-gram 및 TF-IDF 3-gram 벡터화
vectorizer_ast = CountVectorizer(ngram_range=(3, 3))
X_ast_count = vectorizer_ast.fit_transform(all_texts).toarray()
vectorizer_tfidf = TfidfVectorizer(ngram_range=(3, 3))
X_ast_tfidf = vectorizer_tfidf.fit_transform(all_texts).toarray()
vectorizer_simple = CountVectorizer()
X_ast_simple = vectorizer_simple.fit_transform(all_texts).toarray()

# Precision-Recall Curve를 추가하는 함수
def add_precision_recall_curve(X, y, model, label):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(f"\n{label} - Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Class 1만을 위한 Precision-Recall Curve 그리기
    precision, recall, _ = precision_recall_curve(y_test == 1, y_proba[:, 1])
    avg_precision = average_precision_score(y_test == 1, y_proba[:, 1])
    plt.plot(recall, precision, label=f'{label} (AP={avg_precision:.2f})', linewidth=2.5)

# byte histogram 데이터를 위한 CNN 모델
def train_cnn_model():
    img_size = (16, 16)
    normal_dir_1 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/Github/freqeuncy/Github"
    obfuscated_normal_dir_1 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/invokeCradleCrafter"
    obfuscated_normal_dir_2 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/InvokeObfuscation"
    obfuscated_normal_dir_3 = "D:/PowerShell/Extraction/byte histogram_24.10.30/normal/freqeuncy/IseSteroids"
    attack_dir_1 = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_decoded"
    attack_dir_2 = "D:/PowerShell/Extraction/byte histogram_24.10.30/attack/freqeuncy/attack_original"

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

    # 데이터 로드 및 결합
    normal_images_1, normal_labels_1 = load_images_from_folder(normal_dir_1, img_size, label=0)
    obfuscated_normal_images_1, obfuscated_normal_labels_1 = load_images_from_folder(obfuscated_normal_dir_1, img_size, label=1)
    obfuscated_normal_images_2, obfuscated_normal_labels_2 = load_images_from_folder(obfuscated_normal_dir_2, img_size, label=1)
    obfuscated_normal_images_3, obfuscated_normal_labels_3 = load_images_from_folder(obfuscated_normal_dir_3, img_size, label=1)
    attack_images_1, attack_labels_1 = load_images_from_folder(attack_dir_1, img_size, label=2)
    attack_images_2, attack_labels_2 = load_images_from_folder(attack_dir_2, img_size, label=3)

    # 데이터 결합 및 정규화
    X = np.concatenate((normal_images_1, obfuscated_normal_images_1, obfuscated_normal_images_2, obfuscated_normal_images_3, attack_images_1, attack_images_2), axis=0) / 255.0
    y = np.concatenate((normal_labels_1, obfuscated_normal_labels_1, obfuscated_normal_labels_2, obfuscated_normal_labels_3, attack_labels_1, attack_labels_2), axis=0)

    # 학습 및 테스트 데이터 분할
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

    # Precision-Recall Curve 그리기
    y_pred = model.predict(X_test)
    y_proba = y_pred[:, 1]  # Class 1 확률
    precision, recall, _ = precision_recall_curve(y_test == 1, y_proba)
    avg_precision = average_precision_score(y_test == 1, y_proba)
    plt.plot(recall, precision, label=f'CNN Byte Histogram (AP={avg_precision:.2f})', linewidth=2.5)

# 모든 모델과 벡터화 방법에 대해 Precision-Recall Curve 추가
plt.figure(figsize=(10, 8))

# 각 모델과 벡터화 방법의 Class 1에 대한 Precision-Recall Curve 추가
add_precision_recall_curve(X_ast_count, y, RandomForestClassifier(n_estimators=100, random_state=42), "AST 3-gram RF")
add_precision_recall_curve(X_ast_tfidf, y, RandomForestClassifier(n_estimators=100, random_state=42), "AST TF-IDF 3-gram RF")
add_precision_recall_curve(X_ast_simple, y, RandomForestClassifier(n_estimators=100, random_state=42), "AST Random Forest")
add_precision_recall_curve(X_ast_simple, y, GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), "AST Gradient Boosting")

# CNN 모델을 학습시키고, Class 1에 대한 Precision-Recall Curve 추가
train_cnn_model()

# 그래프 설정 - 글자 크기와 선 굵기 조정
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Combined Precision-Recall Curves for Class 1', fontsize=16)

# 범례 및 선 굵기 옵션 적용
plt.legend(loc='best', fontsize=12)
for line in plt.gca().lines:
    line.set_linewidth(2)  # 모든 선의 굵기 설정

# 그래프 표시
plt.grid(True)
plt.show()
