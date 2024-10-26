# AstMetricsToCsv.py
# ExtractAstMetrics.ps1의 결과(.json)을 .csv로 변환
import pandas as pd
import json

# PowerShell 스크립트가 생성한 JSON 파일 경로
json_file_path = r"C:\Users\...\ast_analysis_results.json"

# JSON 파일 읽기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# DataFrame 생성
df = pd.DataFrame(data)

# CSV 파일로 저장(경로지정 및 저장될 파일 명 설정)
csv_file_path = r"C:\Users\...\AstFeatures.csv"
df.to_csv(csv_file_path, index=False)

# 저장 파일 경로 출력
print(f"CSV File Created: {csv_file_path}")
