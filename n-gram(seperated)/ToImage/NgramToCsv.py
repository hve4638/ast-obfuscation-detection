#Ngram을 one-hot-encoding하여 .csv로 변환
import os
import numpy as np
import csv

# AST 노드를 숫자로 매핑하는 딕셔너리
ast_node_map = {
    "PipelineAst": 1,
    "StringConstantExpressionAst": 2,
    "CommandExpressionAst": 3,
    "NamedBlockAst": 4,
    "CommandAst": 5,
    "ScriptBlockAst": 6,
    "ArrayLiteralAst": 7,
    "ErrorExpressionAst": 8
}

# one-hot 인코딩 함수
def one_hot_encode(node_idx, num_classes=8):
    one_hot = [0] * num_classes
    if 1 <= node_idx <= num_classes:
        one_hot[node_idx - 1] = 1
    return one_hot

# nGramResult.txt 파일을 읽어서 처리하는 예시 코드
file_path = r'C:\Users\...t\NgramResult.txt'  # 파일 경로

# CSV 파일이 저장될 디렉토리 설정
output_dir = r'C:\Users\...\csv_files'  # .csv 파일이 저장될 경로

# 디렉토리 없을 시 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 각 .ps1 파일의 n-gram을 저장할 딕셔너리 초기화
ps1_ngrams = {}

# 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # .ps1 파일 이름과 n-gram 구분
        parts = line.split(':')
        if len(parts) > 1:
            ps1_file = parts[0]  # .ps1 파일 이름
            ngram = parts[1].strip()  # n-gram 내용

            # AST 노드를 숫자로 변환
            ngram_nodes = ngram.replace("(", "").replace(")", "").split(",")
            mapped_ngram = [ast_node_map.get(node.strip(), 0) for node in ngram_nodes]  # 매핑되지 않은 노드는 0으로 설정

            # 해당 파일 이름에 n-gram 추가
            if ps1_file not in ps1_ngrams:
                ps1_ngrams[ps1_file] = []
            ps1_ngrams[ps1_file].append(mapped_ngram)

# 각 파일에 대해 one-hot 인코딩된 벡터들을 CSV로 저장
for ps1_file, ngrams in ps1_ngrams.items():
    all_vectors = []
    
    # 각 3-gram에 대해 one-hot 인코딩된 벡터 생성
    for ngram in ngrams:
        for node_idx in ngram:
            if node_idx > 0:  # 매핑된 노드만 처리
                one_hot_vector = one_hot_encode(node_idx)
                all_vectors.append(one_hot_vector)
    
    # numpy 배열로 변환
    vectors_array = np.array(all_vectors)
    
    # CSV 파일로 저장
    csv_filename = f"{ps1_file.replace('.ps1', '')}_onehot.csv"
    csv_file_path = os.path.join(output_dir, csv_filename)  # 지정된 경로로 저장
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(vectors_array)
    
    print(f"{csv_file_path} 파일로 one-hot 인코딩 벡터 저장 완료.")
