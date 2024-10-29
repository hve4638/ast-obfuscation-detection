param(
	[String]$path,
	[String]$pathcsv,
  [String]$dir_name,
  [String]$pathtotal
)
# 깊이 정보를 담을 배열
$results = @()


# .ps1을 포함하는 모든 하위 파일 리스트
$scriptArr = Get-ChildItem -Path $path -Name -Recurse -Include *.ps1
$cnt = 0

# 각 파일에 대하여
foreach ($sc in $scriptArr){
  # 각 파일의 경로
  $dirpath = $path + $sc
  # ast 추출
  $ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $dirpath), [ref]$null, [ref]$null)
  # null 검사
  if($null -ne $ast) {
    # 파일 저장 형식
    $npath = $pathcsv + $dir_name + '_' + $cnt + '.csv'
    # 특성 추출하여 .csv로 저장
    $pro = .\GetProperty -path $dirpath | Export-Csv -path $npath -NoTypeInformation
    # 각 파일에 대한 깊이 정보
    $result = .\DepthCal2 -path $dirpath -filename $sc
    $cnt += 1
  }
  # 깊이 정보 배열에 추가
  $results += $result
}
# 파일 저장 형식 지정
$p = $pathtotal + $dir_name+'.csv'
# 깊이 정보 .csv 파일 생성
# $results | Export-Csv -path $p -NoTypeInformation