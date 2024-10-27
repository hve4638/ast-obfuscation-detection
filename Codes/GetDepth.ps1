param (
	[String]$path
)

$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

$hierarchy = @{}
$min_depth = 0
$max_depth = 0
$mean = 0

$ast.FindAll( { $true }, $true) |
ForEach-Object {
	if ($null -eq $_.Parent) {
		$id = 0
	}
	else {
		$id = $_.Parent.GetHashCode()
	}
	# 해시 테이블에 부모 id key가 존재하지 않으면 새로운 배열 리스트 생성
	if ($hierarchy.ContainsKey($id) -eq $false) {
		$hierarchy[$id] = [System.Collections.ArrayList]@()
	}
	$null = $hierarchy[$id].Add($_)
}

function get-depth($Id, $Indent = 0) {
	if ($min_depth > $Indent) $min_depth = $Indent
	if ($max_depth < $Indent) $max_depth = $Indent
	mean += $Indent
	$hierarchy[$id] | ForEach-Object {
		# '{0}[{1}]: {2}' -f $space, $_.GetType().Name, $_.Extent
		$newid = $_.GetHashCode()
		if ($hierarchy.ContainsKey($newid)) {
			get-depth -id $newid -indent ($indent + 1)
		}
}

mean = mean / hierarchy.lengh