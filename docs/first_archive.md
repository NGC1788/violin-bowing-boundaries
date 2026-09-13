# 첫 공개 아카이브 확인

설치와 GPU 실행 점검을 통과한 다음 단계다. 첫 파일은 Part 1의 Type A sample 1로 정한다. 목적은 원본 파일 구조, 신호 단위·시간축, 상태 판정 근거를 확인하는 것이다. 이것 하나로 악기·현·세션 일반화 평가를 완료하지 않는다.

**2026-09-13 업데이트:** 실제 헤더에서 비압축 크기 69.085 GiB와 번호형 CSV 구조를 확인했다. 오늘 처음 진행한다면 다운로드·검증·보호 조건을 둔 추출을 묶은 [오늘 서버 실행 순서](today_server.md)를 따른다. 아래는 다운로드와 목록 조회를 따로 수행할 때의 참고 명령이다.

## 1. 파일 한 개 다운로드

저장소 폴더에서 실행한다. 확인한 버전 ID는 17749111이다.

```bash
bash scripts/run.sh download \
  --record 17749111 \
  --file '2024-03-25_TypeA_sample1.7z' \
  --max-gib 14
```

조회한 압축 크기는 14,083,546,513 bytes (13.116 GiB), MD5는 `2709034cc10344f37f3488376ae3ebcd`다. 명령은 공식 API에서 현재 파일 정보를 다시 읽고 크기 상한과 체크섬을 확인한다. [공식 버전 레코드](https://zenodo.org/records/17749111).

`Downloaded and MD5 verified`가 나와야 다음으로 진행한다. 동일한 검증 완료 파일이 이미 있으면 `Already present and MD5 verified`가 나온다. 연결이 끊기면 같은 명령으로 재개한다. 이 명령은 압축을 해제하지 않는다.

## 2. 목록과 비압축 크기 확인

`7z`가 있으면 아래 수동 목록 명령을 사용할 수 있다. sudo가 없거나 압축 도구가 없으면 [오늘 서버 실행 순서](today_server.md)의 `setup-tools`를 사용한다. 프로젝트에 설치된 `7zz`는 `run.sh`가 자동으로 찾아 사용한다.

```bash
command -v 7z || command -v 7zz
```

관리자 권한 없는 설치와 목록 검사:

```bash
bash scripts/run.sh setup-tools
bash scripts/run.sh archive-audit \
  --archive data/raw/zenodo/17749111/2024-03-25_TypeA_sample1.7z
```

일반 목록과 상세 목록을 파일로 저장한다.

```bash
mkdir -p reports/data
7z l data/raw/zenodo/17749111/2024-03-25_TypeA_sample1.7z > reports/data/typeA_sample1_listing.txt
7z l -slt data/raw/zenodo/17749111/2024-03-25_TypeA_sample1.7z > reports/data/typeA_sample1_technical.txt
```

각 명령이 오류 없이 끝났는지 확인하고, 요약을 읽는다.

```bash
head -n 35 reports/data/typeA_sample1_listing.txt
tail -n 8 reports/data/typeA_sample1_listing.txt
head -n 35 reports/data/typeA_sample1_technical.txt
df -h .
```

일반 목록 끝의 비압축 총량과 파일 수, 상세 목록 헤더의 `Solid`·`Blocks`·`Physical Size`를 확인한다. 압축파일 크기만으로 전체 해제 공간을 추정하지 않는다. 부분 해제를 하더라도 압축 블록 구성에 따라 상당한 처리가 필요할 수 있다.

목록 검사와 보고서 생성만 자동으로 수행하려면 다음을 사용한다.

```bash
bash scripts/run.sh archive-audit \
  --archive data/raw/zenodo/17749111/2024-03-25_TypeA_sample1.7z
```

목록 검사를 통과하고 해제 공간이 충분할 때의 전체 추출은 `prepare-first --extract`에 구현되어 있다. [오늘 서버 실행 순서](today_server.md)의 조건과 명령을 사용한다. 완료된 다운로드는 MD5를 다시 확인한 후 재사용한다. CSV 열 수·단위·타임스탬프·샘플 수를 검증하기 전에 모델 학습을 시작하지 않는다.
