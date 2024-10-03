param (
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