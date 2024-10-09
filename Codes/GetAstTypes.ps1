# 경로를 받아오는 매개변수
param (
	[String]$path
)

# Parser를 이용해 AST 추출
$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

$predicate = { $true }

$recurse = $true

# AST 노드 위에서부터 탐색 후 객체 타입 추출
$astnodeTypes = $ast.FindAll($predicate, $recurse) | ForEach-Object {$_.GetType().Name}

# AST 타입 리스트 반환
return $astnodeTypes
