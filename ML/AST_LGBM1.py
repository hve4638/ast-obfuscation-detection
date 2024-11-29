import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# AST 파일들을 읽어들여 리스트로 반환하는 함수
def read_ast_files(directory):
    ast_texts = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            ast_texts.append(file.read())
    return ast_texts

# 디렉토리로부터 AST 파일들을 읽어들이는 함수
def gather_data_from_directories(directories, label):
    all_texts = []
    all_labels = []
    for directory in directories:
        ast_texts = read_ast_files(directory)
        all_texts.extend(ast_texts)
        all_labels.extend([label] * len(ast_texts))
    return all_texts, all_labels

# 경로 설정
# 정상 난독화 데이터 경로
obfuscated_benign_ast_dirs = [
    'D:/PowerShell/Extraction/InvokeCradleCrafter/InvokeCradleCrafter(ast)',
    'D:/PowerShell/Extraction/InvokeObfuscation/InvokeObfuscation(ast)'
]

# 악성 난독화 데이터 경로
obfuscated_malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/encoded/mentor(ast)'

# 데이터 읽어오기
obfuscated_benign_texts, obfuscated_benign_labels = gather_data_from_directories(obfuscated_benign_ast_dirs, label=0)
obfuscated_malicious_texts, obfuscated_malicious_labels = gather_data_from_directories([obfuscated_malicious_ast_dir], label=1)

# 전체 데이터 결합
X_texts = obfuscated_benign_texts + obfuscated_malicious_texts
y_labels = obfuscated_benign_labels + obfuscated_malicious_labels

# 벡터화 (모든 데이터를 한 번에 처리)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_texts).toarray()
y = np.array(y_labels)

# K-Fold 교차 검증 설정 (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# 교차 검증 수행
for train_index, test_index in kf.split(X):
    print(f"\nFold {fold}:")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # LightGBM 모델 생성 및 학습
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = model.predict(X_test)

    # 결과 평가
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    fold += 1
