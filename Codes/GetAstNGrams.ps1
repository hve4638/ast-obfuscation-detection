param (
    [String]$path,
    [String]$path1,
    [String]$path2,
    [String]$path3,
    [String]$path4,
    [String]$dir_name
)

# 하위 파일 중 .ps1을 포함하는 파일 모두 가져옴
$scriptArr = Get-ChildItem -Path $path -Name -Recurse -Include *.ps1
# 파일 번호
$cnt = 0

foreach ($sc in $scriptArr){
    $dirpath = $path + $sc
    # Ast 노드 타입 추출
    $nodeTypes = .\GetAstTypes -path $dirpath
    # Ast 노드 타입 문자열이 null이 아니면 n-gram생성
    if ($null -ne $nodeTypes) {
        # n-gram 생성
        $Ast2Grams = .\GetNGrams -InputArr $nodeTypes -N 2
        $Ast3Grams = .\GetNGrams -InputArr $nodeTypes -N 3
        $Ast4Grams = .\GetNGrams -InputArr $nodeTypes -N 4
        # 저장 경로, 형식 설정
        $npath1 = $path1 + $dir_name + '_' + $cnt + '_AST.txt' # Ast 저장 경로
        $npath2 = $path2 + $dir_name + '_' + $cnt + '_2gram.txt' # 2-gram 저장 경로
        $npath3 = $path3 + $dir_name + '_' + $cnt + '_3gram.txt' # 3-gram 저장 경로
        $npath4 = $path4 + $dir_name + '_' + $cnt + '_4gram.txt' # 4-gram 저장 경로
        # 파일로 저장
        Set-Content -Path $npath1 -Value $nodeTypes
        Set-Content -Path $npath2 -Value $Ast2Grams
        Set-Content -Path $npath3 -Value $Ast3Grams
        Set-Content -Path $npath4 -Value $Ast4Grams
        $cnt += 1
    }
}