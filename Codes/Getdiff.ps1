param(
    [String]$path,
    [String]$pathcsv
)

# AST 파싱
$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)
    
$results = @()
$visitedNodes = New-Object System.Collections.Generic.HashSet[object]     # 중복 방지를 위한 해시셋

$totalLeafNodes = 0
$x_diff = 0
$leafDepth = -1s


# AST 트리를 재귀적으로 순환하며 특성 추출
function Calculate-ASTMetrics {
    param ($node, $currentDepth)

    # 노드를 이미 방문했는지 확인
    if ($visitedNodes.Contains($node)) {
       return
    }
    [void]$visitedNodes.Add($node)  # 방문 기록 추가 (반환값-True 무시)

    if ($currentDepth -gt $script:leafDepth){
        $script:leafDepth = $currentDepth
        $script:x_diff--
    }
    elseif($script:leafDepth -ge $currentDepth){
        $script:x_diff += 2    
        # 리프 노드 빠져나온 경우 깊이 초기화
        if ($script:leafDepth -gt $currentDepth){
            $script:leafDepth = 0
            $script:x_diff += 4
        }
    }

    # Ast 특성 변수
    if($currentDepth -eq 0){
        $result = [PSCustomObject]@{
        Type = $node.GetType().Name
        Depth = $currentDepth
        Parent = $null
        x_diff = 0
        y_diff = 0
        Code = $node.Extent.Text
    }
    }
    else {
        $result = [PSCustomObject]@{
        Type = $node.GetType().Name
        Depth = $currentDepth
        Parent = $node.Parent.GetType().Name
        x_diff = $script:x_diff
        y_diff = $currentDepth * -1
        Code = $node.Extent.Text
    }
    }
    
    # 특성 배열에 변수 추가
    $script:results += $result

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

return $results | Export-Csv -path $pathcsv -NoTypeInformation
