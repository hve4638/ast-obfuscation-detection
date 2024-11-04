import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 4-gram AST 파일들을 읽어들여 리스트로 반환하는 함수
def read_4gram_files(directory):
    ngram_texts = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            ngram_texts.append(file.read())
    return ngram_texts

# 디렉토리로부터 4-gram 파일들을 읽어들이는 함수
def gather_data_from_directories(directories, label):
    all_texts = []
    all_labels = []
    for directory in directories:
        ngram_texts = read_4gram_files(directory)
        all_texts.extend(ngram_texts)
        all_labels.extend([label] * len(ngram_texts))
    return all_texts, all_labels

# 경로 설정
benign_ast_dir = 'D:/PowerShell/Extraction/GithubGist/GithubGist(4gram)'

obfuscated_benign_ast_dirs = [
    'D:/PowerShell/Extraction/InvokeCradleCrafter/InvokeCradleCrafter(4gram)',
    'D:/PowerShell/Extraction/InvokeObfuscation/InvokeObfuscation(4gram)',
    'D:/PowerShell/Extraction/IseSteroids/IseSteroids(4gram)'
]

malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/decoded/decoded(4gram)'

obfuscated_malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/encoded/mentor(4gram)'

# 데이터 읽어오기
benign_texts, benign_labels = gather_data_from_directories([benign_ast_dir], label=0)
obfuscated_benign_texts, obfuscated_benign_labels = gather_data_from_directories(obfuscated_benign_ast_dirs, label=1)
malicious_texts, malicious_labels = gather_data_from_directories([malicious_ast_dir], label=2)
obfuscated_malicious_texts, obfuscated_malicious_labels = gather_data_from_directories([obfuscated_malicious_ast_dir], label=3)

# TF-IDF 벡터화 (4-gram 데이터를 처리)
vectorizer = TfidfVectorizer(ngram_range=(4, 4))  # 4-gram 기반으로 벡터 변환
X = vectorizer.fit_transform(benign_texts + obfuscated_benign_texts + malicious_texts + obfuscated_malicious_texts).toarray()
y = np.array(benign_labels + obfuscated_benign_labels + malicious_labels + obfuscated_malicious_labels)

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
