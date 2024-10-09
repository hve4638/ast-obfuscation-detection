# AST, N-Gram 추출 코드 설명

각 코드들에 관한 설명 및 호출 방법

## GetAst.ps1

\$path의 Ast 노드를 위에서부터 탐색하여 노드 내용 출력

호출
```bash
GetAst -path $path
```

## GetAstObjects.ps1

\$path의 Ast 노드를 위에서부터 탐색하여 노드 타입, 코드 위치, 코드 출력

호출
```bash
GetAstObjects -path $path
```

## GetVisualizedAst.ps1

\$path의 Ast 노드를 시각화하여 출력

호출
```bash
GetVisualizedAst -path $path
```

## Ast 탐색 UI 모듈

```bash
# 다운로드
Install-Module -Name ShowPSAst -Scope CurrenUser -Force

# 로드
Show-Ast -InputObject $path
```

## GetAstTypes.ps1

\$path의 Ast 노드를 위에서부터 탐색하여 노드 타입 반환

호출
```bash
GetAstTypes -path $path
```

## GetNGrams.ps1

Ast 노드 타입 리스트, N을 매개변수로 받아 N-gram 형식으로 반환

호출 
```bash
# 3-gram
GetNGrams -InputArr $InputArr -N 3
# 4-gram
GetNGrams -InputArr $InputArr -N 4
```

## GetAstTypes, GetNGrams를 이용해 Ast, n-gram 추출하여 저장하는 전체 코드

```bash
$path = # 절대 경로 or 상대 경로 입력

$nodeTypes = GetAstTypes -path $path

$Ast3Grams = GetNGrams -InputArr $nodeTypes -N 3
$Ast4Grams = GetNGrams -InputArr $nodeTypes -N 4

$path3 = # Ast3Gram을 저장할 경로
$path4 = # Ast4Gram을 저장할 경로

Set-Content -Path $path3 -Value $Ast3Grams
Set-Content -Path $path4 -Value $Ast4Grams
```

## 디렉토리 전체 파일의 Ast, n-gram 추출하여 저장하기
```bash
# 경로 설정 (마지막에 \ 붙일것!)
$path = # 디렉토리 경로
$path1 = # Ast 저장 경로
$path2 = # 2-gram 저장 경로
$path3 = # 3-gram 저장 경로
$path4 = # 4-gram 저장 경로

# path 디렉토리의 전체 파일의 3,4-gram을 각각path3, path4에 저장
GetAstNGrams -path $path -path1 $path1 -path2 $path2 -path3 $path3 -path4 $path4
```