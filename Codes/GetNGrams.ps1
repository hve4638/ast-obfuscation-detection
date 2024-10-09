# AST 타입 리스트, N을 받아오는 매개변수
param (
	[string[]]$InputArr,
	[int]$N
)

# AST 타입 리스트 확인
if (-not $InputArr) {
	Write-Error "InputArr is empty."
	return @()
}

# 결과를 저장할 빈 배열
$nGrams = @()

# InputArr로부터 N-Gram 추출
for ($i = 0; $i -le $InputArr.Length - $N; $i++) {
	$nGram = $InputArr[$i..($i + $N - 1)] -join ','
	$nGrams += "($nGram)"
}

return $nGramsparam (
	[string[]]$InputArr,
	[int]$N
)
if (-not $InputArr) {
	Write-Error "InputArr is empty."
	return @()
}
$nGrams = @()

for ($i = 0; $i -le $InputArr.Length - $N; $i++) {
	$nGram = $InputArr[$i..($i + $N - 1)] -join ','
	$nGrams += "($nGram)"
}
return $nGrams
