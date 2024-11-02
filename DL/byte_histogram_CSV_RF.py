import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ast

# CSV 파일 불러오기 (경로는 각 파일의 경로로 수정)
#normal_github_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/normal_origianl/normal_Github/entropy.csv')
powershellgallery_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/normal_origianl/PowerShellGallery/entropy.csv')
invoke_cradle_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/normal/invokeCradleCrafter/entropy.csv')
invoke_obfuscation_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/normal/InvokeObfuscation/entropy.csv')
isesteroids_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/normal/IseSteroids/entropy.csv')
attack_base64decoded_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/attack_base64decoded/entropy.csv')
attack_original_df = pd.read_csv('D:/PowerShell/Extraction/byte histogram/attack_original/entropy.csv')

# 각 데이터프레임에 라벨을 추가 (정상: 0, 정상 난독화: 1, 악성: 2, 악성 난독화: 3)
#normal_github_df['label'] = 0
powershellgallery_df['label'] = 0
invoke_cradle_df['label'] = 1
invoke_obfuscation_df['label'] = 1
isesteroids_df['label'] = 1
attack_base64decoded_df['label'] = 2
attack_original_df['label'] = 3

# 데이터 병합
df = pd.concat([#normal_github_df,
                powershellgallery_df, invoke_cradle_df, invoke_obfuscation_df, isesteroids_df,
                attack_base64decoded_df, attack_original_df])

# 'bytes' 열의 값을 리스트로 변환
df['bytes'] = df['bytes'].apply(lambda x: ast.literal_eval(x))

# Feature와 Label 분리
X = pd.DataFrame(df['bytes'].to_list())  # 'bytes' 열을 리스트로 변환 후 데이터프레임으로 변환
y = df['label']

# 데이터 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터셋을 훈련/테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 테스트 세트로 예측 수행
y_pred = rf_model.predict(X_test)

# 성능 평가
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
