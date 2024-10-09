#Datasets To n-gram
#설명 : 파워쉘 데이터셋이 있는 폴더 내에 있는 모든 .ps1의 AST, 3-gram 생성 후 출력 + 지정한 경로에 result.txt로 저장해줌
#스크립트 파싱 및 AST생성과정 포함
#1. 데이터셋 경로 지정, 2. result.txt경로 지정해주면 됨.(결과파일 존재하면 기존 파일은 삭제됨)

# 1. n-gram을 생성하는 함수
function Get-NGrams {
    param (
        [string[]]$InputArr,  # 입력 배열
        [int]$N = 3           # n-gram 크기
    )

    if (-not $InputArr) {
        Write-Error "InputArr is empty."
        return @()
    }

    $nGrams = @()

    for ($i = 0; $i -le $InputArr.Length - $N; $i++) {
        # 연속된 N개의 AST 노드를 쉼표로 결합하여 하나의 n-gram으로 만듦
        $nGram = $InputArr[$i..($i + $N - 1)] -join ','
        $nGrams += "($nGram)"
    }

    return $nGrams
}

# 2. 폴더 내 모든 .ps1 파일에 대해 AST를 추출하고 n-gram 생성
function Process-ScriptsInFolder {
    param (
        [string]$FolderPath,  # 폴더 경로
        [string]$OutputFile,  # 결과가 저장될 파일 경로
        [int]$N = 3           # n-gram 크기
    )

    # 폴더 내 모든 .ps1 파일 목록 가져오기
    $scriptFiles = Get-ChildItem -Path $FolderPath -Filter *.ps1

    foreach ($file in $scriptFiles) {
        Write-Host "Processing file: $($file.FullName)"
        
        # 파일 내용을 읽음
        $scriptContent = Get-Content -Path $file.FullName -Raw

        # 스크립트를 파싱하여 AST 추출
        $ast = [System.Management.Automation.Language.Parser]::ParseInput($scriptContent, [ref]$null, [ref]$null)

        # AST에서 노드 타입 추출
        $nodeTypes = $ast.FindAll({$true}, $true) | ForEach-Object { $_.GetType().Name }

        # n-gram 생성
        $nGrams = Get-NGrams -InputArr $nodeTypes -N $N

        # 파일별로 n-gram 출력 및 파일에 저장 
        $nGrams | ForEach-Object { 
            "$($file.Name): $_" | Add-Content -Path $OutputFile #함수 내부에서 생성되지만, 외부로 전달되도록 해야 값이 저장됨(result.txt)
        }
    }
}

# 3. 폴더 경로 및 결과 파일 경로 설정
$folderPath = 'C:\Users\데이터셋 경로지정'
$outputFile = 'C:\Users\result.txt 경로지정(담길 폴더 생성 후 지정)' #ex) c:\Users\abc\Desktop\outputfile

# 4. 결과 파일이 존재하면 삭제 (중복 방지)
if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

# 5. n-gram 분석 실행
Process-ScriptsInFolder -FolderPath $folderPath -OutputFile $outputFile -N 3