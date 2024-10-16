import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

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
malicious_obfuscated_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/mentor(ast)'

benign_obfuscated_ast_dirs = [
    'D:/PowerShell/Extraction/InvokeCradleCrafter/InvokeCradleCrafter(ast)',
    'D:/PowerShell/Extraction/InvokeObfuscation/InvokeObfuscation(ast)',
    'D:/PowerShell/Extraction/IseSteroids/IseSteroids(ast)'
]

# 데이터 읽어오기
benign_texts, benign_labels = gather_data_from_directories(benign_obfuscated_ast_dirs, label=0)
malicious_texts, malicious_labels = gather_data_from_directories([malicious_obfuscated_ast_dir], label=1)

# 벡터화 (모든 데이터를 한 번에 처리)
vectorizer = CountVectorizer()  # AST 노드를 벡터로 변환
X = vectorizer.fit_transform(benign_texts + malicious_texts).toarray()
y = np.array(benign_labels + malicious_labels)

# 학습용과 테스트용 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 결과 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
