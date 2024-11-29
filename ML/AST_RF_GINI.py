import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# CSV 파일에서 'B2'와 'C2'부터 데이터를 읽어와 텍스트로 결합하는 함수
def read_csv_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                # 첫 번째 행을 건너뛰고 B, C 열 데이터 사용
                df = pd.read_csv(filepath, skiprows=[0])

                # B2와 C2부터 실제 데이터 시작, B와 C 열의 값을 텍스트로 결합
                type_column = df.iloc[:, 1].astype(str)  # 두 번째 열 (B 열)
                parents_column = df.iloc[:, 2].astype(str)  # 세 번째 열 (C 열)
                combined_text = ' '.join(type_column + ' ' + parents_column)

                if combined_text.strip():  # 텍스트가 비어 있지 않으면 추가
                    data.append(combined_text)
            except Exception as e:
                print(f"Error in file {filename}: {e}")
    return data

# 디렉토리로부터 CSV 파일들을 읽어들이는 함수
def gather_data_from_directories(directories, label):
    all_texts = []
    all_labels = []
    for directory in directories:
        texts = read_csv_files(directory)
        all_texts.extend(texts)
        all_labels.extend([label] * len(texts))
    return all_texts, all_labels

# 경로 설정
obfuscated_benign_ast_dirs = [
    'D:/PowerShell/Extraction/CSV/InvokeCradleCrafter',
    'D:/PowerShell/Extraction/CSV/InvokeObfuscation',
    'D:/PowerShell/Extraction/CSV/IseSteroids'
]
obfuscated_malicious_ast_dir = 'D:/PowerShell/Extraction/CSV/Encoded_Powershell/Mentor_Obj_Feature(Obfuscated)'

# TensorFlow 및 GPU 확인
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))

# 전체 시간 측정 시작
total_start_time = time.time()

# 데이터 읽어오기
obfuscated_benign_texts, obfuscated_benign_labels = gather_data_from_directories(obfuscated_benign_ast_dirs, label=0)
obfuscated_malicious_texts, obfuscated_malicious_labels = gather_data_from_directories([obfuscated_malicious_ast_dir], label=1)

# 전체 데이터 결합 및 비어있는 데이터 필터링
X_texts = [text for text in obfuscated_benign_texts + obfuscated_malicious_texts if text.strip()]
y_labels = obfuscated_benign_labels + obfuscated_malicious_labels[:len(X_texts)]

# 데이터가 비어 있는지 최종 확인
if not X_texts:
    raise ValueError("Error: X_texts is empty. Please check the data sources or file structure.")

# 벡터화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_texts).toarray()
y = np.array(y_labels)

# 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 학습 시간 측정 시작
train_start_time = time.time()

# RandomForestClassifier 모델 학습
model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
model.fit(X_train, y_train)

# 학습 시간 측정 종료
train_end_time = time.time()

# 예측 및 평가
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# 전체 시간 측정 종료
total_end_time = time.time()

# 시간 출력
print(f"Total Time: {total_end_time - total_start_time:.4f} seconds")
print(f"Training Time: {train_end_time - train_start_time:.4f} seconds")
