# 컬렉션 전체 처리

2026-09-17. 공개 로봇 운궁 컬렉션(Zenodo Part 1–8, 압축 17개, 약 250 GiB)을 서버에서 **하루 이틀 동안 무인으로** 처리한다.

## 무엇을 하나

압축 파일마다 다음을 반복한다.

1. 내려받고 MD5를 확인한다(기존 `download`와 같은 검증). Zenodo가 연결마다 속도를 제한하므로(서버 측정: 연결 1개 0.73 MiB/s, 연결 8개를 더하면 합계 4.72 MiB/s) 기본으로 **연결 8개**를 쓴다(`--connections`). 64 MiB 조각마다 정확한 범위 응답만 받아들이고, 끝난 조각은 `.part.chunks.json`에 기록해 끊겨도 그 조각부터 이어받는다. 예전 방식으로 받던 `.part`도 이미 받은 부분을 그대로 이어받는다. `--delete-archive`를 주지 않으면 압축 파일은 남는다.
2. 7-Zip 목록에서 속도 폴더(`v1`, `v2`, `2024-03-27_r_v1` 등)를 찾는다. 한 폴더의 부모가 Schelleng 격자 하나다(예: `rep_1`, `rep_2`, 현 종류별 폴더).
3. **속도 폴더 하나만** 풀고, 파일 이름·크기를 목록과 대조한다.
4. `trial-audit`로 검사하면서 분석 창(4열 모두)을 `data/cache/<격자>/<폴더>/window_N.npy`(float32)에, 활 힘·속도의 전체 궤적을 1 kHz 평균으로 `profile_N.npy`에 저장한다.
5. 푼 CSV를 지운다. 원본은 압축 파일에 그대로 있다.
6. 한 격자의 폴더가 모두 끝나면 캐시로 `regime-map`을 돌린다. 판정·그림·flyback 법칙·Schelleng 법칙이 `reports/collection/<격자>/regime_map/`에 저장된다.

상태는 `reports/collection/state/<압축파일>.json`에 저장되므로 **끊겨도 같은 명령을 다시 실행하면 이어서 한다.** 다음 압축 파일은 디스크 여유가 있을 때만 미리 받는다.

## 멈추기

`pkill -TERM -f "scripts/collection.py run"`으로 멈춘다. 받던 조각은 다음 실행에서 다시 받는다. 남은 다운로드 잠금 파일은, 그 프로세스가 끝났으면 다음 실행이 자동으로 치운다.

## 삭제 규칙

- 푼 CSV(`data/staging/`)는 캐시가 모두 저장된 뒤, 또는 실패한 뒤에만 지운다.
- 압축 파일은 `--delete-archive`를 줬을 때, 그 파일의 **모든 격자가 성공한 뒤에만** 지운다. 실패한 압축 파일은 남긴다.
- 이미 풀어 둔 `data/interim/typeA_sample1`은 그 자리에서 읽기만 하고 절대 지우지 않는다. 그 압축 파일도 지우지 않는다.
- 7-Zip이 목록과 다른 파일을 내놓으면(구조적 문제) 전체 실행을 멈춘다. 다른 실패(손상된 시행 등)는 그 격자만 실패로 기록하고 다음으로 넘어간다.

## 실행

```bash
bash scripts/run.sh collection plan
bash scripts/run.sh collection plan --list data/raw/zenodo/17749111/2024-03-25_TypeA_sample1.7z --probe
nohup bash scripts/run.sh collection run --workers "$(( $(nproc) - 2 ))" --delete-archive > logs/collection.log 2>&1 < /dev/null &
tail -f logs/collection.log | grep --line-buffered -E "^\[|COLLECTION|REGIME MAP|FAIL|Error|Traceback"
bash scripts/run.sh collection status
```

`--only 파일이름`으로 일부만 처리할 수 있다. 순서는 현 종류(A1, B1, C1, D1) → 설치 반복(Part 6) → 같은 종류의 다른 현(A2, A3) → 힘 4–12 N(Part 7, 8) → 나머지다.

## 디스크

- 속도 폴더 하나를 풀면 17–24 GiB를 쓴다.
- 압축 파일은 13–29 GiB다.
- 캐시는 전체 약 40 GiB다(시행당 약 320 KB).
- 여유가 모자라면 그 격자는 "needs … GiB" 메시지와 함께 실패로 기록된다. 공간을 만든 뒤 다시 실행하면 이어서 한다.
- A1 처리가 끝난 뒤 `data/interim/typeA_sample1`(69 GiB)을 지우면 여유가 크게 생긴다. 이 삭제는 사용자가 직접 결정한다(캐시와 압축 파일이 남아 있어 다시 만들 수 있다).
