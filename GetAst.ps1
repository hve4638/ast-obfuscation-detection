param(
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

$ast.FindAll($predicate, $recurse)