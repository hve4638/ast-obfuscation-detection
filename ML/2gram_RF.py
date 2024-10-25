import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 2-gram AST 파일들을 읽어들여 리스트로 반환하는 함수
def read_2gram_files(directory):
    ngram_texts = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            ngram_texts.append(file.read())
    return ngram_texts

# 디렉토리로부터 2-gram 파일들을 읽어들이는 함수
def gather_data_from_directories(directories, label):
    all_texts = []
    all_labels = []
    for directory in directories:
        ngram_texts = read_2gram_files(directory)
        all_texts.extend(ngram_texts)
        all_labels.extend([label] * len(ngram_texts))
    return all_texts, all_labels

# 경로 설정
benign_ast_dir = 'D:/PowerShell/Extraction/GithubGist/GithubGist(2gram)'

obfuscated_benign_ast_dirs = [
    'D:/PowerShell/Extraction/InvokeCradleCrafter/InvokeCradleCrafter(2gram)',
    'D:/PowerShell/Extraction/InvokeObfuscation/InvokeObfuscation(2gram)',
    'D:/PowerShell/Extraction/IseSteroids/IseSteroids(2gram)'
]

malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/decoded/decoded(2gram)'

obfuscated_malicious_ast_dir = 'D:/PowerShell/Extraction/mentor_dataset/encoded/mentor(2gram)'

# 데이터 읽어오기
benign_texts, benign_labels = gather_data_from_directories([benign_ast_dir], label=0)
obfuscated_benign_texts, obfuscated_benign_labels = gather_data_from_directories(obfuscated_benign_ast_dirs, label=1)
malicious_texts, malicious_labels = gather_data_from_directories([malicious_ast_dir], label=2)
obfuscated_malicious_texts, obfuscated_malicious_labels = gather_data_from_directories([obfuscated_malicious_ast_dir], label=3)

# 벡터화 (모든 데이터를 한 번에 처리)
vectorizer = CountVectorizer()  # 2-gram 데이터를 벡터로 변환
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
