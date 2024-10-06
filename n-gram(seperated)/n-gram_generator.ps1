#파워쉘 하나에 대한 n-gram생성 및 출력
#스크립트 파싱 후 AST생성 후 사용.
# n-gram을 생성하는 함수
function Get-NGrams {
    param (
        [string[]]$InputArr,  # 입력 배열
        [int]$N = 3           # n-gram 크기
    )
    if (-not $InputArr) {
        Write-Error "InputArr is empty."
        return @()
    }
    $nGrams = @()
    for ($i = 0; $i -le $InputArr.Length - $N; $i++) {
        # 연속된 N개의 AST 노드를 쉼표로 결합하여 하나의 n-gram으로 만듦
        $nGram = $InputArr[$i..($i + $N - 1)] -join ','
        $nGrams += "($nGram)"
    }

    return $nGrams
}
# $ast가 이미 존재하므로 AST 노드 타입을 바로 추출
$nodeTypes = $ast.FindAll({$true}, $true) | ForEach-Object { $_.GetType().Name }
# $ast가 없을 경우
#$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content "C:\path\script.ps1" -Raw), [ref]$null)

# 3-gram 생성
$nGrams = Get-NGrams -InputArr $nodeTypes -N 3
# 결과 출력
$nGrams | ForEach-Object { Write-Output $_ }