# AST-Obfuscation-Detection

24F KU 캡스톤 디자인 프로젝트 : *AST기반 난독화 및 악성 스크립트 탐지*

## 참가자

- 박예은
- 배강민
- 양희도
- 이한빈
- 조리노

## 규칙

### 1. 데이터셋, 이미지 등을 업로드하지 말 것

대신 공유된 `Onedrive` 폴더에 올려주세요

### 2. 모든 작업은 개인 branch에서 진행

각자 작업의 충돌 방지를 위해 직접 main branch에 commit 하지 말기 바랍니다

branch 이름은 `dev-식별자-(필요시추가)` 를 권장합니다 (*ex. "dev-hanbin"*)

필요시 여러개의 branch를 활용하세요

**branch 확인**

```bash
# branch 목록을 확인
git branch

# 현재 가르키는 branch를 확인
git status
```

**branch 생성 및 이동**

```bash
# branch 생성 (생성한 branch로 바로 이동하는 것이 아님)
git branch 새브랜치

# 해당 branch 이동
git switch 새브랜치
```

**branch 병합 (필요시)**

```bash
# main(기본) branch 이동
git switch main

# (새브랜치)을 main branch에 반영함
git merge 새브랜치 --no-ff -m "커밋 메시지"
```

### 3. 커밋 메시지 규칙

- 특정 프로그램, 기능에 관련된 경우 앞에 해당 프로그램명을 적어주세요. (*ex. "[ASTTreeParser] 기능 추가"*)

- 그 외 정해진 형식은 없습니다

### 4. git reset 사용 금지

대신 `git revert`를 사용하세요
