# 바이올린 원음 미래 예측: 고정12개 실험 완료·STOP

**2026-11-02 최종 결과:** 고정12개 학습과 단일 시험 평가를 완료했다. 개발 자료로 선택한Markov의 시험log loss는0.552413,제안offset은0.664192로20.23%높았다.2%개선screen실패,추가탐색없이STOP이다. [전체결과](RESULTS.md), [봉인점수](RESULTS.json), [최종감사](FINAL_AUDIT.json), [그림](RESULTS_FIGURE.png)을읽는다. 아래는준비단계이력이다. 완료된train/evaluate를다시실행하지않는다. 실행코드·모델·지표도추가동결되었다.

2026-11-02 현재 `PREPARED_SUMMARY.json`이 집계 증거다.2,168녹음×60개 입력을 만들었으며 참여자34/8/9명(train/dev/test), 녹음1442/337/389개다. 실제 미래 변조8건의 입력 bitwise 동일성을 확인했다. 시험 정답은 생성됐으나 모델 학습·선택·점수 계산에는 사용하지 않았다. 바이올린 모델 성능 결과는 아직 없다.

- `protocol.json`, `PROTOCOL.md`: 실측 목표를 만들기 전에 정한 방법·비교군·지표·예산. 데이터 동결에 포함되어 수정하지 않는다.
- `DATA_REVIEW_INITIAL.json`, `DATA_REVIEW.json`: Claude Opus4.7의 서로 다른 도구 비활성 호출로 수행한 초기/의존 코드 보완 검토. 둘 다 data-preparation scope 승인이고 원시 CLI 출력은 비공개다. 이것은 인간 심사나 독립적인 경험적 확인을 뜻하지 않는다.
- `data_freeze_manifest.json`: 데이터 준비용 소스·규약·검토 해시. 이후 학습/평가 실행기는 별도 검토와 동결이 필요하다. 데이터 준비 코드를 수정하거나 같은 경로로 재실행하여 덮어쓰지 않는다.
- `metrics.py`, `test_metrics.py`: 참여자 macro 점수와 paired bootstrap의 합성 검증. 실제 평가 실행기는 아직 없다.
- `LITERATURE_SCOPE_20261102.md`: offset·CDF·coherence·제약 학습의 선행 범위. 방법론의 높은 신규성은 아직 입증되지 않았다.

원음과 개별 metadata는 `~/violin-rne/data/external/blanco2021_raw/`, 준비 자료는 이 폴더의 ignored `local/prepared/`다. `features.npz`에는 입력/과거 상태/분할 ID만 있다. 미래 Y9·연속 목표·guard는 `train_labels.npz`, `dev_labels.npz`, `test_labels.npz`로 분리했다. 파일 내용 hash는 공개 summary에 있지만 특징·정답·녹음·참여자 목록은 공개하지 않는다.

준비 명령 `python -m agentic_research.violin_raw_v1.prepare`는 이미 완료됐으며, 같은 폴더에서 다시 실행하면 중단하도록 했다. `download.py`도 영수증을 다시 쓰므로 지금 재실행하지 않는다. 데이터 소스/준비 hash 검증부터 이어간다. 다른 컴퓨터에서 같은 실행을 무작정 덮어쓰기보다 원음·의존 라이브러리를 확인하고 별도 재현 계약과 경로를 만든다. macOS libarchive/bsdtar를 사용한 아카이브 감사는 GNU tar만 있는 Linux에 그대로 이식됐다고 보장하지 않는다.

다음 순서는 고정12개 학습·예측 저장·봉인 → dev로 단순 비교군 결정·봉인 → test1회 평가를 구현하는 실행기와 그 별도 검토다. 기준은 `protocol.json`이며, 더 좋은 숫자가 나올 때까지 모델/seed/문턱을 바꾸지 않는다. 학습 입력은 train X/Y와 prefix 상태로 제한하고 test 정답 파일을 열지 않는다. 동일 참여자의 여러 녹음과3seed를 추가 독립 표본으로 세지 않는다.

초기 메타데이터 감사의 `inventory_summary.json`에는AG2 미분류로 gate=false가 남아 있다. 원래 CG/FG/EG 허용 목록을 유지한 해소 기록은 `data_partition_receipt.json`이다. 실제 대상 녹음을 바꾸어 통과시킨 것이 아니며 원래 실패 기록을 보존했다.
