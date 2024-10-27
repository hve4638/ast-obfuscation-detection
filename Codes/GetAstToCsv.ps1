param(
	[String]$path,
	[String]$pathcsv,
  [String]$dir_name,
  [String]$pathtotal
)

$results = @()

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
$parent = @{
    Name = 'Parents'
    Expression = { $_.Parent.GetType().Name }
}
$depth = @{
    Name = 'Depth'
    Expression = {  }
}

$scriptArr = Get-ChildItem -Path $path -Name -Recurse -Include *.ps1
$cnt = 0


foreach ($sc in $scriptArr){
    $dirpath = $path + $sc
    $ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $dirpath), [ref]$null, [ref]$null)
    if($null -ne $ast) {
        $npath = $pathcsv + $dir_name + '_' + $cnt + '.csv'
        $astObjects = $ast.FindAll($predicate, $recurse)
        $astObjects | Select-Object -Property $position, $type, $parent, $text | Export-Csv -Path $npath -NoTypeInformation
        $result = .\DepthCal -path $dirpath -filename $sc
        $cnt += 1
    }
    $results += $result
}

$p = $pathtotal + $dir_name+'.csv'
$results | Export-Csv -path $p -NoTypeInformation