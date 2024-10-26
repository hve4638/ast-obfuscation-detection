# ExtractAstMetrics.ps1
# 폴더에 있는 모든 PowerShell 파일의 AST의 freature 4가지(생성 노드수, 노드 종류수, 최대 깊이, 평균 깊이)를 추출하여 .json으로 저장하는 스크립트

# 추출을 진행할 데이터셋 폴더 경로 지정
param (
    [String]$folderPath = "C:\Users\...\Datasets"
)

# 폴더 내 모든 .ps1 파일 가져오기
$files = Get-ChildItem -Path $folderPath -Filter *.ps1

# 결과 저장을 위한 배열 생성
$results = @()

foreach ($file in $files) {
    # AST 추출
    $ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $file.FullName), [ref]$null, [ref]$null)

    # 노드 정보 계산
    $totalNodes = 0
    $totalDepth = 0
    $maxDepth = 0
    $uniqueNodeTypes = New-Object System.Collections.Generic.HashSet[string]
    $visitedNodes = New-Object System.Collections.Generic.HashSet[object]

    function Calculate-ASTMetrics {
        param ($node, $currentDepth)

        if ($visitedNodes.Contains($node)) {
            return
        }
        $visitedNodes.Add($node) | Out-Null

        $script:totalNodes++
        $script:totalDepth += $currentDepth
        [void]$uniqueNodeTypes.Add($node.GetType().Name)

        if ($currentDepth -gt $script:maxDepth) {
            $script:maxDepth = $currentDepth
        }

        foreach ($property in $node.GetType().GetProperties()) {
            $childNode = $property.GetValue($node)
            if ($childNode -is [System.Management.Automation.Language.Ast]) {
                Calculate-ASTMetrics -node $childNode -currentDepth ($currentDepth + 1)
            } elseif ($childNode -is [System.Collections.IEnumerable]) {
                foreach ($item in $childNode) {
                    if ($item -is [System.Management.Automation.Language.Ast]) {
                        Calculate-ASTMetrics -node $item -currentDepth ($currentDepth + 1)
                    }
                }
            }
        }
    }

    # 초기 호출
    Calculate-ASTMetrics -node $ast -currentDepth 1

    # 평균 깊이 계산
    $averageDepth = if ($totalNodes -gt 0) { [math]::Round($totalDepth / $totalNodes, 2) } else { 0 }

    # 결과 저장 (파일 이름을 첫 번째 열로 설정, 나머진 순서대로)
    $result = [PSCustomObject]@{
        FileName       = $file.Name
        TotalNodes     = $totalNodes
        NodeTypes      = $uniqueNodeTypes.Count
        MaxDepth       = $maxDepth
        AverageDepth   = $averageDepth
    }

    $results += $result
}

# 출력 디렉터리 확인 및 생성
$outputDirectory = "C:\Users\...\output"
if (-not (Test-Path -Path $outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory
}

# 결과를 JSON 파일로 저장 (저장될 .json파일 명 지정 ex.ast_analysis_result.json)
$outputFilePath = "$outputDirectory\ast_analysis_results.json"
$results | ConvertTo-Json | Set-Content -Path $outputFilePath

# 저장된 파일 경로 출력
Write-Output "AST Analysis JSON Created: $outputFilePath"
