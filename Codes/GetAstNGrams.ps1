param (
    [String]$path,
    [String]$path1,
    [String]$path2,
    [String]$path3,
    [String]$path4
)

$scriptArr = Get-ChildItem -Path $path -Name

foreach ($sc in $scriptArr){
    $dirpath = $path + $sc
    $nodeTypes = .\GetAstTypes -path $dirpath
    $Ast2Grams = .\GetNGrams -InputArr $nodeTypes -N 2
    $Ast3Grams = .\GetNGrams -InputArr $nodeTypes -N 3
    $Ast4Grams = .\GetNGrams -InputArr $nodeTypes -N 4
    $npath1 = $path1 + $sc + '.txt' # Ast 저장 경로
    $npath2 = $path2 + $sc + '.txt' # 2-gram 저장 경로
    $npath3 = $path3 + $sc + '.txt' # 3-gram 저장 경로
    $npath4 = $path4 + $sc + '.txt' # 4-gram 저장 경로
    Set-Content -Path $npath1 -Value $nodeTypes
    Set-Content -Path $npath2 -Value $Ast2Grams
    Set-Content -Path $npath3 -Value $Ast3Grams
    Set-Content -Path $npath4 -Value $Ast4Grams
}