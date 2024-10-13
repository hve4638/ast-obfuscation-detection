# DatasetsToNgram_log.ps1 실행하여 txt파일 생성 후 그 txt파일을 분석
# AST 노드를 숫자로 매핑하는 딕셔너리 => ps1 데이터셋 분석후 종류 추가 가능
ast_node_map = {
    "PipelineAst": 1,
    "StringConstantExpressionAst" : 2,
    "CommandExpressionAst" : 3,
    "NamedBlockAst" : 4,
    "CommandAst" : 5,
    "ScriptBlockAst" : 6,
    "ArrayLiteralAst" : 7,
    "ErrorExpressionAst" : 8
}

# nGramResult.txt 파일을 읽어서 처리하는 예시 코드
file_path = r'C:\...\NgramResult.txt'  #분석할 .txt 파일 경로
# 각 .ps1 파일의 n-gram을 저장할 딕셔너리 초기화
ps1_ngrams = {}
unused_nodes = {}  # 사용되지 않은 노드를 저장할 딕셔너리
new_nodes = {}  # 매핑되지 않은 새로운 노드를 저장할 딕셔너리
all_used_nodes = {}  # 모든 파일에서 사용된 AST 노드를 추적할 딕셔너리 (매핑된 노드: 빈도수)
unmapped_nodes = {}  # 매핑되지 않은 노드를 추적할 딕셔너리 (매핑되지 않은 노드: 빈도수)
file_unmapped_nodes = {}  # 각 파일에서 매핑되지 않은 노드들을 추적

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
            mapped_ngram = [str(ast_node_map.get(node.strip(), node.strip())) for node in ngram_nodes]

            # 해당 파일 이름에 n-gram 추가
            if ps1_file not in ps1_ngrams:
                ps1_ngrams[ps1_file] = []
            ps1_ngrams[ps1_file].append(mapped_ngram)

            # 해당 파일에서 사용된 AST 노드를 추적하기 위한 집합 생성
            used_nodes = set(node.strip() for node in ngram_nodes)

            # 매핑된 노드 추적
            for node in used_nodes:
                if node in ast_node_map:  # 매핑된 노드
                    if node in all_used_nodes:
                        all_used_nodes[node] += 1
                    else:
                        all_used_nodes[node] = 1
                else:  # 매핑되지 않은 노드 추적
                    if node in unmapped_nodes:
                        unmapped_nodes[node] += 1
                    else:
                        unmapped_nodes[node] = 1
                    
                    # 파일별 매핑되지 않은 노드 추적
                    if ps1_file not in file_unmapped_nodes:
                        file_unmapped_nodes[ps1_file] = set()
                    file_unmapped_nodes[ps1_file].add(node)

            # 사용되지 않은 노드 추적
            if ps1_file not in unused_nodes:
                unused_nodes[ps1_file] = set(ast_node_map.keys())  # 모든 노드를 미리 저장
            unused_nodes[ps1_file] -= used_nodes  # 사용된 노드를 제거

            # 새로운 노드 추적 (매핑되지 않은 노드)
            if ps1_file not in new_nodes:
                new_nodes[ps1_file] = set()  # 새로운 노드를 저장할 집합
            for node in ngram_nodes:
                if node.strip() not in ast_node_map:
                    new_nodes[ps1_file].add(node.strip())  # 매핑되지 않은 노드를 추가

# 결과 출력
for ps1_file, ngrams in ps1_ngrams.items():
    print(f"\n{ps1_file} 파일의 매핑된 n-grams:")
    for ngram in ngrams:
        print(f"- {', '.join(ngram)}")
    
    # 사용되지 않은 노드 출력
    print(f"\n{ps1_file} 파일에서 사용되지 않은 노드:")
    for unused_node in unused_nodes[ps1_file]:
        print(f"- {unused_node}")

    # 리스트에 없었던 새로운 노드 출력
    print(f"\n{ps1_file} 파일에서 발견된 새로운 노드:")
    for new_node in new_nodes[ps1_file]:
        print(f"- {new_node}")

# 모든 파일에서 사용된 매핑된 AST 노드 리스트와 빈도수 출력
sorted_mapped_nodes_by_frequency = sorted(all_used_nodes.items(), key=lambda x: x[1], reverse=True)

print(f"\n모든 파일에서 존재했던 매핑된 AST 노드 리스트: {len(all_used_nodes)}개")
for node, count in sorted_mapped_nodes_by_frequency:
    print(f"- {node} (빈도수: {count})")

# 매핑되지 않은 노드도 포함한 모든 노드 리스트 출력 (빈도수 포함)
print(f"\n모든 파일에서 1번이라도 발생한 노드 리스트 (매핑되지 않은 노드 포함): {len(all_used_nodes) + len(unmapped_nodes)}개")
for node, count in sorted_mapped_nodes_by_frequency:
    print(f"- {node} (빈도수: {count})")

# 매핑되지 않은 노드도 출력
sorted_unmapped_nodes_by_frequency = sorted(unmapped_nodes.items(), key=lambda x: x[1], reverse=True)
for node, count in sorted_unmapped_nodes_by_frequency:
    print(f"- {node} (빈도수: {count})")

# 추가 기능: 매핑된 AST 노드 리스트에 없었던 노드들이 어떤 파일에서 발생했는지 출력
print("\n매핑되지 않은 노드들이 발생한 파일 리스트:")
for ps1_file, nodes in file_unmapped_nodes.items():
    node_list = ', '.join(nodes)
    print(f"{ps1_file} 파일에서 발견된 매핑되지 않은 노드: {node_list}")
