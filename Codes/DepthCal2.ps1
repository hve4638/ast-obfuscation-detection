# DeptCalc.ps1
# 파워셸 스크립트에서(1개) AST를 탐색하여 4가지 특성을 추출(Total Nodes, Node Types, Max Depth, Average Depth)

# 파워쉘 스크립트 경로 지정
param(
    [String]$path,
    [String]$filename
)

# AST 파싱
$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

# AST 트리 탐색에 필요한 변수들
$totalNodes = 0                       # 전체 노드 수
$totalLeafNodes = 0                   # 전체 리프 노드 수
$totalDepth = 0                       # 전체 깊이 (평균 깊이 계산에 사용)
$maxDepth = 0                         # 리프 노드 최대 깊이
$minDepth = 100                       # 리프 노드 최소 깊이
$leafDepth = -1                       # 리프 노드의 깊이
$totalLeafDepth = 0                   # 리프노드 전체 깊이 (평균 깊이 계산에 사용)
$uniqueNodeTypes = New-Object System.Collections.Generic.HashSet[string]  # 고유 노드 타입을 추적
$visitedNodes = New-Object System.Collections.Generic.HashSet[object]     # 중복 방지를 위한 해시셋

# AST 트리를 재귀적으로 순회하며 특성 값을 계산하는 함수
function Calculate-ASTMetrics {
    param ($node, $currentDepth)

    # 노드를 이미 방문했는지 확인
    if ($visitedNodes.Contains($node)) {
        return
    }
    [void]$visitedNodes.Add($node)  # 방문 기록 추가 (반환값-True 무시)

    # 노드 수와 깊이 합산
    $script:totalNodes++
    $script:totalDepth += $currentDepth

    # 현재 노드의 타입을 고유 노드 타입에 추가
    [void]$uniqueNodeTypes.Add($node.GetType().Name)  # 반환 값 무시 - [void]

    # 디버깅을 위한 노드 타입 및 깊이 출력
    # Write-Output "Node Type: $($node.GetType().Name), Current Depth: $currentDepth"

    # 최대 깊이 갱신
    if ($currentDepth -gt $script:maxDepth) {
        $script:maxDepth = $currentDepth
    }
    # 
    if ($currentDepth -gt $script:leafDepth){
        $script:leafDepth = $currentDepth
    }
    elseif($script:leafDepth -ge $currentDepth){
        $script:totalLeafDepth +=$script:leafDepth
        $script:totalLeafNodes++
        # Write-Output "LeafDepth: $script:leafDepth"
        # Write-Output "totalLeafDepth: $script:totalLeafDepth"
        if($script:leafDepth -lt $script:minDepth){
            $script:minDepth = $script:leafDepth
            # Write-Output "minDepth: $script:minDepth"
        }
        if ($script:leafDepth -gt $currentDepth){
            $script:leafDepth = 0
        }
    }

    # 자식 노드 탐색
    if($null -eq $node){
        return
    }
    foreach ($property in $node.GetType().GetProperties()) {
        $childNode = $property.GetValue($node)
        if ($childNode -is [System.Management.Automation.Language.Ast]) {
            Calculate-ASTMetrics -node $childNode -currentDepth ($currentDepth + 1)
        }
        elseif ($childNode -is [System.Collections.IEnumerable]) {
            foreach ($item in $childNode) {
                if ($item -is [System.Management.Automation.Language.Ast]) {
                    Calculate-ASTMetrics -node $item -currentDepth ($currentDepth + 1)
                }
            }
        }
    }

}

# 초기 호출
Calculate-ASTMetrics -node $ast -currentDepth 0

# 마지막 리프 노드 계산
    $script:totalLeafDepth +=$script:leafDepth
    $script:totalLeafNodes++
    # Write-Output "LeafDepth: $script:leafDepth"
    # Write-Output "totalLeafDepth: $script:totalLeafDepth"
    if($script:minDepth -gt $script:leafDepth){
        $script:minDepth = $script:leafDepth
        # Write-Output "minDeapth: $minDepth"
    }

# 평균 깊이 계산
$averageDepth = if ($totalNodes -gt 0) { [math]::Round($totalDepth / $totalNodes, 2) } else { 0 }
#
$averageLeafDepth =  if ($totalLeafNodes -gt 0) { [math]::Round($totalLeafDepth / $totalLeafNodes, 2) } else { 0 }

# 결과 출력
<#
Write-Output "Total Nodes: $totalNodes"
Write-Output "Total Leaf Nodes: $totalLeafNodes"
Write-Output "Node Types: $($uniqueNodeTypes.Count)"
Write-Output "Max Depth: $maxDepth"
Write-Output "Min Depth: $minDepth"
Write-Output "Average Depth: $averageDepth"
Write-Output "Total Leaf Depth: $totalLeafDepth"
Write-Output "Average Leaf Depth: $averageLeafDepth"
#>
$result = [PSCustomObject]@{
    Filename       = $filename
    TotalNodes     = $totalNodes
    TotalLeafNodes = $totalLeafNodes
    NodeTypes      = $uniqueNodeTypes.Count
    MaxDepth       = $maxDepth
    MinDepth       = $minDepth
    AverageDepth   = $averageDepth
    AverageLeafDepth = $averageLeafDepth
}

return $result