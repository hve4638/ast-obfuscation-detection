import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 모든 파일의 'Type' 고유 값을 수집하여 원-핫 인코더를 초기화하는 함수
def initialize_encoder(directories):
    types = set()
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                ast_data = pd.read_csv(file_path, names=['Type', 'Depth', 'Parent', 'Code'], delimiter=None, header=0)
                if 'Type' in ast_data.columns:
                    types.update(ast_data['Type'].dropna().unique())
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(list(types)).reshape(-1, 1))
    return encoder

# 각 ps1 파일을 하나의 샘플로 변환하는 함수
def load_ast_data_from_directories(directories, encoder, label):
    data = []
    labels = []

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                ast_data = pd.read_csv(file_path, names=['Type', 'Depth', 'Parent', 'Code'], delimiter=None, header=0)

                if ast_data.empty or 'Type' not in ast_data.columns:
                    print(f"Skipping empty or invalid file: {file_path}")
                    continue

                # 'Type' 원-핫 인코딩
                type_encoded = encoder.transform(ast_data[['Type']])

                # Depth와 Parent 존재 여부를 수치형으로 변환
                depths = ast_data['Depth'].values.reshape(-1, 1)
                parent_present = ast_data['Parent'].notna().astype(int).values.reshape(-1, 1)

                # 피처 결합: Type(원-핫 인코딩), Depth, Parent 존재 여부
                features = np.hstack((type_encoded, depths, parent_present))

                # 각 ps1 파일에 대해 평균 벡터를 만들어 하나의 샘플로 요약
                summarized_features = features.mean(axis=0)

                data.append(summarized_features)
                labels.append(label)

    return np.array(data), np.array(labels)

# 경로 설정 (정상 난독화, 악성 난독화 데이터셋)
obfuscated_normal_dirs = [
    "D:/PowerShell/CSV/2/InvokeCradleCrafter",
    "D:/PowerShell/CSV/2/InvokeObfuscation",
    "D:/PowerShell/CSV/2/IseSteroids"
]

obfuscated_malicious_dirs = [
    "D:/PowerShell/CSV/2/EncodedPowershell/Mentor_Obj_Feature(Obfuscated)"
]

# 모든 데이터를 대상으로 원-핫 인코더 초기화
encoder = initialize_encoder(obfuscated_normal_dirs + obfuscated_malicious_dirs)

# 데이터 로드 및 라벨링
normal_data, normal_labels = load_ast_data_from_directories(obfuscated_normal_dirs, encoder, label=0)  # 정상 난독화: 라벨 0
malicious_data, malicious_labels = load_ast_data_from_directories(obfuscated_malicious_dirs, encoder, label=1)  # 악성 난독화: 라벨 1

# 데이터 병합 및 라벨링
X = np.vstack((normal_data, malicious_data))
y = np.concatenate((normal_labels, malicious_labels))

# KFold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracy_scores = []

for train_index, test_index in kf.split(X):
    print(f"\nFold {fold}")

    # 학습 및 테스트 데이터 분할
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # LightGBM 모델 생성 및 학습
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 피처 중요도 출력
    importances = model.feature_importances_
    feature_names = encoder.get_feature_names_out(['Type']).tolist() + ['Depth', 'Parent_Present']

    # 피처 중요도와 이름 매핑하여 출력
    for feature, importance in zip(feature_names, importances):
        print(f"Feature: {feature}, Importance: {importance}")

    # 예측 및 성능 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    fold += 1

# 전체 Fold의 평균 성능 출력
print("\nAverage Accuracy across 5 folds:", np.mean(accuracy_scores))
