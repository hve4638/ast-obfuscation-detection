# $target = "./dataset/attack/8c5d.ps1"
# $target = "../dataset/normal/Get.ps1"
$target = $args[0]

$ast = [System.Management.Automation.Language.Parser]::ParseFile($target, [ref]$null, [ref]$null)

function extractVariables {
    $variable = $AST.FindAll( { $args[0] -is [System.Management.Automation.Language.VariableExpressionAst ] }, $true)
    $variable = $variable | Where-Object { $_.parent.left -or $_.parent.type -and ($_.parent.operator -eq 'Equals' -or $_.parent.parent.operator -eq 'Equals') }
    $variable | Select-Object Extent
}
function extractCommand {
    $commands = $AST.FindAll( { $args[0] -is [System.Management.Automation.Language.CommandAst ] }, $true)
    $commands | Select-Object Extent
}

function extractAll {
    $finded = $AST.FindAll({$true}, $true)
    $finded | Select-Object Extent
    # | StaticType
}

extractAll $ast