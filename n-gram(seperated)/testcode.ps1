#testcode for AST

# 스크립트를 변수에 로드
$scriptPath = 'C:\Path\To\Script.ps1'
$scriptContent = Get-Content -Path $scriptPath -Raw

# 스크립트 내용을 파싱하여 AST를 얻음
$ast = [System.Management.Automation.Language.Parser]::ParseInput($scriptContent, [ref]$null, [ref]$null)

# AST에서 모든 함수 정의 찾기
$functionDefinitions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

# 함수 이름 표시
$functionDefinitions | ForEach-Object { $_.Name }

# AST 트리 내의 모든 노드를 순회하며 출력
$ast.FindAll({ $true }, $true) | ForEach-Object {
    $_.GetType().Name + ": " + $_.Extent.Text
}
# AST에서 함수 정의만 찾기
$functionDefinitions = $ast.FindAll({ $_ -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

# 각 함수 정의의 이름 출력
$functionDefinitions | ForEach-Object {
    $_.Name
}