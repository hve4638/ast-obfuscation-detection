param (
	[String]$path
)

$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

$hierarchy = @{}

$ast.FindAll( { $true }, $true) |
ForEach-Object {
	if ($null -eq $_.Parent) {
		$id = 0
	}
	else {
		$id = $_.Parent.GetHashCode()
	}
	if ($hierarchy.ContainsKey($id) -eq $false) {
		$hierarchy[$id] = [System.Collections.ArrayList]@()
	}
	$null = $hierarchy[$id].Add($_)
}

function Visualize-Tree($Id, $Indent = 0) {

	$space = '--' * $indent
	$hierarchy[$id] | ForEach-Object {
		'{0}[{1}]: {2}' -f $space, $_.GetType().Name, $_.Extent
	
		$newid = $_.GetHashCode()
		if ($hierarchy.ContainsKey($newid)) {
			Visualize-Tree -id $newid -indent ($indent + 1)
		}
	}
}

Visualize-Tree -id 0
