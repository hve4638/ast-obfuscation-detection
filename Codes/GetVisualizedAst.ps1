# 경로를 받아오는 매개변수
param (
	[String]$path
)

# Parser를 이용해 AST 추출
$ast = [System.Management.Automation.Language.Parser]::ParseInput((Get-Content $path), [ref]$null, [ref]$null)

# 해시 테이블 생성
$hierarchy = @{}

# AST 노드 위에서부터 탐색
$ast.FindAll( { $true }, $true) |
ForEach-Object {
	# Parent가 널인 경우: 루트 노드인 경우
	if ($null -eq $_.Parent) {
		$id = 0
	}
	# Parent의 해시 코드를 id에 저장
	else {
		$id = $_.Parent.GetHashCode()
	}
	# 해시 테이블에 부모 id key가 존재하지 않으면 새로운 배열 리스트 생성
	if ($hierarchy.ContainsKey($id) -eq $false) {
		$hierarchy[$id] = [System.Collections.ArrayList]@()
	}
	# 현재 노드를 부모 id key에 추가
	$null = $hierarchy[$id].Add($_)
}

# 시각화를 위한 재귀 함수
function Visualize-Tree($Id, $Indent = 0) {
	# 현재 트리 레벨 만큼 문자열 출력
	$space = '--' * $indent
	# 현재 id에 대한 노드 탐색
	$hierarchy[$id] | ForEach-Object {
		# 트리 레벨, 노드 타입, 코드의 위치와 범위 출력
		'{0}[{1}]: {2}' -f $space, $_.GetType().Name, $_.Extent

		# 자신의 해시 코드 저장
		$newid = $_.GetHashCode()
		# 자식 노드 탐색
		if ($hierarchy.ContainsKey($newid)) {
			Visualize-Tree -id $newid -indent ($indent + 1)
		}
	}
}

# 루트 노드부터 함수 호출
Visualize-Tree -id 0