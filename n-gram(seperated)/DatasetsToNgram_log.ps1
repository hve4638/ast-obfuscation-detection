#DatasetsToNgram_log.ps1
#DatasetsToNgram과 동일하게 result(n-gram).txt생성 + log.txt생성(기능 추가 및 함수 역할 분리)
#사용시 1. 데이터셋 경로 지정, 2. result(n-gram).txt 경로, 3. log.txt 경로 설정해야 함.(하단)

# 1. 스크립트 내용을 파싱하여 AST를 추출하는 함수
function Get-AST {
    param (
        [string]$scriptContent,  # 스크립트 내용 (텍스트)
        [string]$logFile         # 로그 파일 경로
    )

    # 스크립트를 파싱하여 AST 추출
    try {
        $ast = [System.Management.Automation.Language.Parser]::ParseInput($scriptContent, [ref]$null, [ref]$null)
        Add-Content -Path $logFile -Value "AST parsed successfully"
    } catch {
        Add-Content -Path $logFile -Value "Error parsing AST: $_"
        return $null
    }
    return $ast
}

# 2. AST에서 노드 타입을 추출하는 함수
function Extract-ASTNodeTypes {
    param (
        [System.Management.Automation.Language.Ast]$ast,  # AST 객체
        [string]$logFile                                  # 로그 파일 경로
    )

    try {
        # AST에서 노드 타입 추출
        $nodeTypes = $ast.FindAll({$true}, $true) | ForEach-Object { $_.GetType().Name }
        $nodeTypeCount = $nodeTypes.Count
        Add-Content -Path $logFile -Value "AST node types extracted: $nodeTypeCount types ($($nodeTypes -join ', '))"
        return $nodeTypes
    } catch {
        Add-Content -Path $logFile -Value "Error extracting AST node types: $_"
        return @()
    }
}

# 3. n-gram을 생성하는 함수
function Get-NGrams {
    param (
        [string[]]$InputArr,  # 입력 배열 (노드 타입 배열)
        [int]$N = 3,          # n-gram 크기
        [string]$logFile      # 로그 파일 경로
    )

    if (-not $InputArr) {
        Add-Content -Path $logFile -Value "InputArr is empty."
        return @()
    }

    $nGrams = @()

    for ($i = 0; $i -le $InputArr.Length - $N; $i++) {
        # 연속된 N개의 AST 노드를 쉼표로 결합하여 하나의 n-gram으로 만듦
        $nGram = $InputArr[$i..($i + $N - 1)] -join ','
        $nGrams += "($nGram)"
    }

    Add-Content -Path $logFile -Value "[n-gram generated]: $($nGrams.Count) n-grams"
    return $nGrams
}

# 4. 폴더 내 모든 .ps1 파일을 처리하는 함수
function Process-ScriptsInFolder {
    param (
        [string]$FolderPath,      # 폴더 경로
        [string]$OutputFile,      # 결과가 저장될 파일 경로
        [string]$LogFile,         # 로그 파일 경로
        [int]$N = 3               # n-gram 크기
    )

    # 폴더 내 모든 .ps1 파일 목록 가져오기
    $scriptFiles = Get-ChildItem -Path $FolderPath -Filter *.ps1

    foreach ($file in $scriptFiles) {
        Add-Content -Path $LogFile -Value "[Processing file : $($file.Name)]"
        
        # 파일 내용을 읽음
        try {
            $scriptContent = Get-Content -Path $file.FullName -Raw
            Add-Content -Path $LogFile -Value "-------------------------------------------------------------File content loaded--------------------------------------------------------------"
        } catch {
            Add-Content -Path $LogFile -Value "[Error loading file content!!!]: $_"
            continue
        }

        # 1. 스크립트 내용을 파싱하여 AST 추출
        $ast = Get-AST -scriptContent $scriptContent -logFile $LogFile
        if (-not $ast) { continue }

        # 2. AST에서 노드 타입 추출
        $nodeTypes = Extract-ASTNodeTypes -ast $ast -logFile $LogFile

        # 3. n-gram 생성
        $nGrams = Get-NGrams -InputArr $nodeTypes -N $N -logFile $LogFile

        # 4. 결과를 파일에 저장
        try {
            $nGrams | ForEach-Object { 
                "$($file.Name): $_" | Add-Content -Path $OutputFile 
            }
            Add-Content -Path $LogFile -Value "n-gram data saved for $($file.FullName)"
        } catch {
            Add-Content -Path $LogFile -Value "Error saving n-gram data for $($file.FullName): $_"
        }
         # 파일 처리 끝날 때마다 구분선 추가
        Add-Content -Path $LogFile -Value "====================================================================================="
    }
}

# 5. 폴더 경로 및 결과 파일 경로 설정 후 실행
$folderPath = 'C:\Users\파워쉘 데이터셋 폴더 경로'  # 파워쉘 데이터셋 경로
$outputFile = 'C:\Users\n-gram생성 파일 경로설정'    # 결과 파일 경로
$logFile = 'C:\Users\로그 파일 경로 설정'          # 로그 파일 경로

# 로그 파일이 존재하면 삭제 후 새로 생성
if (Test-Path $logFile) {
    Remove-Item $logFile
}
# 결과 파일이 존재하면 삭제 (중복 방지)
if (Test-Path $outputFile) {
    Remove-Item $outputFile
}
# n-gram 분석 실행
Process-ScriptsInFolder -FolderPath $folderPath -OutputFile $outputFile -LogFile $logFile -N 3
Write-Host "CSV File Path : $outputFile"
Write-Host "Log File Path: $logFile"