# 경로를 받아오는 매개변수
param(
	[String]$path
)

# Parser를 이용해 AST 추출
$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

$predicate = { $true }

$recurse = $true

# 객체 타입 추출
$type = @{
  Name = 'Type'
  Expression = { $_.GetType().Name }
}

# 코드 위치 추출
$position = @{
  Name = 'Position'
  Expression = { '{0,3}-{1,-3}' -f  $_.Extent.StartOffset, $_.Extent.EndOffset, $_.Extent.Text }
}

# 코드 추출
$text = @{
  Name = 'Code'
  Expression = { $_.Extent.Text }
}

# AST 노드 위에서부터 탐색
$astObjects = $ast.FindAll($predicate, $recurse)

# AST 타입, 코드 위치, 코드 출력
$astObjects | Select-Object -Property $position, $type, $text
param (
	[String]$path
)

$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

$predicate = { $true }

$recurse = $true

$type = @{
  Name = 'Type'
  Expression = { $_.GetType().Name }
}

$position = @{
  Name = 'Position'
  Expression = { '{0,3}-{1,-3}' -f  $_.Extent.StartOffset, $_.Extent.EndOffset, $_.Extent.Text }
}

$text = @{
  Name = 'Code'
  Expression = { $_.Extent.Text }
}

$astnodeTypes = $ast.FindAll($predicate, $recurse) | ForEach-Object {$_.GetType().Name}

return $astnodeTypes
