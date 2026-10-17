# Mac 연구 실행 안내 — 2026-10-17 갱신

현재 연구는 Apple Silicon CPU에서 실행한다. GPU 서버, 추가 센서, 자체 바이올린 녹음은 사용하지 않는다. 패키지 업데이트와 Linux CUDA 환경 변경 없이 `requirements-mac.lock.txt`의 별도 환경을 사용한다.

## 설치와 기본 검증

저장소 루트에서 실행한다. `uv`가 필요하며 관리자 권한은 필요하지 않다.

```bash
bash scripts/bootstrap_mac.sh
bash scripts/run_mac.sh test
```

설치 스크립트는 `.venv-mac`를 만들고 Python 3.11.15와 기록된 패키지를 설치한다. 실행 스크립트는 BLAS 기본 스레드를 1개로, Numba를 2개로 제한한다. 물리 실행기는 배치의 예상 배열 용량이 1 GiB를 넘으면 거부한다. 이 제한은 전체 프로세스 메모리를 보장하는 운영체제 제한은 아니다.

## 파생 데이터 벤치마크

```bash
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_A --source A
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_B --source B
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_C --source C
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_D --source D
```

출력 폴더는 비어 있어야 한다. 이미 있는 결과를 덮어쓰지 않는다. 입력은 먼저 과거 `reports/`의 고정 파일을 찾고, 없으면 `research/inputs/`의 압축된 파생 자료를 사용한다. 어느 경우든 원문 CSV의 SHA256이 고정값과 일치해야 한다.

A/0.1 m/s만 사용한 모델이 주 비교다. B/C/D를 학습원으로 한 실험도 관측 정의에는 A 기준 주파수를 유지하므로, 이를 완전히 독립적인 전체 파이프라인 교차검증이라고 부르지 않는다. 네 표본은 이미 과거 연구에서 열어 본 자료다.

```bash
bash scripts/run_mac.sh python scripts/speed_prior_audit.py
bash scripts/run_mac.sh python scripts/boundary_transport.py --help
bash scripts/run_mac.sh python scripts/multispeed_scaling.py --output reports/multispeed_scaling/reproduction
bash scripts/run_mac.sh python scripts/affine_transport.py \
  --cello-baselines reports/multispeed_scaling/reproduction \
  --output reports/affine_transport/reproduction
bash scripts/run_mac.sh python scripts/affine_width_ablation.py \
  --full-run reports/affine_transport/reproduction \
  --output reports/affine_width_ablation/reproduction
bash scripts/run_mac.sh python scripts/twospeed_known_prior.py \
  --flow-results reports/affine_transport/reproduction/results.json \
  --output reports/twospeed_known_prior/reproduction
```

위 실행들은 주 비교 결과를 본 뒤 설계한 후속 진단이다. 새로운 확증 실험인 것처럼 섞지 않는다. 두 속도 비교 → 흐름 모델 → 폭 고정 절제·기존 비대칭 모델 비교 순서로 실행한다. 자세한 결과와 한계는 [흐름 모델 문서](../research/2026-10-13/affine_transport_assessment.md)에 있다. `SVC(probability=True)`는 고정된 scikit-learn 버전에서 동작하지만 향후 폐기 경고가 나온다. 이 경고 때문에 실험 도중 추정기를 바꾸지 않는다.

위 네 학습원 결과와 흐름·폭 고정 결과를 만든 뒤, 같은 관측 기준에서 모델 순위가 뒤집히는지와 경계가 얼마나 식별되는지 감사한다. `--benchmark-base` 아래에 반드시 `benchmark_A`부터 `benchmark_D`까지 네 폴더가 있어야 한다.

```bash
bash scripts/run_mac.sh python scripts/observer_dominance_audit.py \
  --benchmark-base reports/reproduction \
  --full-run reports/affine_transport/reproduction \
  --k0-run reports/affine_width_ablation/reproduction \
  --out reports/observer_dominance/reproduction
bash scripts/run_mac.sh python scripts/source_identifiability_audit.py \
  --flow-run reports/affine_transport/reproduction \
  --output reports/source_identifiability/reproduction
bash scripts/run_mac.sh python scripts/width_law_certificate.py \
  --all-observers --output reports/width_law/reproduction
```

마지막 명령은 명목 제어값의 위치·속도 열에서 공통 폭 변화율이 강제하는 최소 추가 불일치를 계산한다. 예측기를 학습하거나 A2를 사용하지 않는다. 실제 첼로 예측기에 들어가는 측정 위치·속도는 열 안에서도 조금씩 다르므로, 이 값은 그 예측기의 인증된 손실 하한이 아니다. [폭 법칙 진단 결과와 공개 묶음 검증](../research/2026-10-14/width_law_assessment.md)에 모든 186개 조건과 별도 확인 절차가 있다.

공개된 진단 묶음으로 실제 학습 모델과 같은 `k ∈ [−2,2]` 제한의 영향도 재현할 수 있다. 기존 결과를 보존하도록 새 출력 파일을 지정한다.

```bash
mkdir -p reports
bash scripts/run_mac.sh python research/2026-10-14/width_bound_sensitivity.py \
  --output reports/width_bound_replay.json
```

[제한 범위 민감도 결과](../research/2026-10-14/width_bound_sensitivity.md)는 전체 186개 조건과 새로운 경계 구성 증거를 포함한다. 이는 새로운 예측 모델의 성능 비교가 아니다.

## 논문에서 복원한 바이올린 현 자료

원문 PDF 없이 포함된 파생 셀 표로 비교를 실행할 수 있다. 원문은 저작권 때문에 저장소에 재배포하지 않는다. 추출 과정·원문 URL·체크섬은 자료 README에 있다.

```bash
bash scripts/run_mac.sh python research/2026-10-13/schoonderwaldt2008/benchmark_published.py \
  --output reports/published_violin/reproduction
bash scripts/run_mac.sh python research/2026-10-13/plot_published_predictions.py \
  --predictions reports/published_violin/reproduction/predictions.csv \
  --out reports/published_violin/reproduction/comparison
```

원문 판정이 불확실하거나 분석하지 않은 부분은 포함하지 않는다. E현의 남은 셀은 공간적으로 선택된 표본이므로 D현 전체 격자와 같은 성능 지표로 합치지 않는다.

## 물리 모델과 관측 판정 감사

```bash
bash scripts/run_mac.sh python research/2026-10-13/physics_check.py
bash scripts/run_mac.sh python research/2026-10-13/violin_synthetic_benchmark.py
bash scripts/run_mac.sh python research/2026-10-13/convergence_factorial.py
bash scripts/run_mac.sh python research/2026-10-13/delay_method_factorial.py
bash scripts/run_mac.sh python research/2026-10-13/endpoint_observer_audit.py
bash scripts/run_mac.sh python research/2026-10-13/verify_set_bounds.py
```

이는 일반적인 가상 현과 지정된 마찰 법칙의 계산 실험이다. 개방현 주파수가 바이올린과 같다는 사실만으로 실제 악기를 재현하지 않는다. 과거 보정값은 수정된 수동성 조건을 위반할 수 있으므로 재사용할 수 없다. 기존 보정 로그에서 재개하려 하면 프로그램이 오류를 내는 것이 정상이다.

## 완료된 A2 평가: 작은 파생 파일로 재현

두 고정 평가는 **2026-10-14 06:40 UTC(한국 시각 15:40)**에 완료됐다. 원래 강건 후보는 0.635452로 학습원에서 선택한 단순 기준 모델 0.659570에 졌다. 별도 흐름 후보의 주 학습 속도 쌍(0.05/0.10 m/s)도 0.653586으로 폭 고정 0.667870과 학습 지수 0.662273에 졌다. 보조 속도 쌍의 일부 이득은 주 비교 실패를 대체하지 않는다. [완료 결과·출처·전체 파일](../research/2026-10-14/lockbox_A2/README.md)에 26개 모델의 결과를 보존했다.

대형 아카이브 없이도 공개 파생 특징으로 모든 예측·지표를 다시 계산할 수 있다.

```bash
bash scripts/run_mac.sh python scripts/restore_frozen_models.py
bash scripts/run_mac.sh python research/2026-10-14/lockbox_release.py replay
```

첫 명령은 A2를 학습에 사용하지 않고 공개 학습 입력과 고정 파라미터로 원래 14개 체크포인트를 복원한다. 기존 파일이 있으면 해시를 검사하고 다른 파일은 덮어쓰지 않는다. `--verify-rebuild`를 추가하면 기존 파일이 있어도 재적합해 같은 체크섬인지 검사한다. 나머지 12개 후보의 계수는 고정 명세에 포함되어 있다. 두 번째 명령은 공개 파일·코드를 검증하고 저장된 전체 지표를 허용오차 `1e-12`로 대조한다. 이는 **파생 특징 이후의 예측·평가 재현**이며, 원시 파형 전처리나 없는 아카이브의 MD5 검사를 대신하지 않는다.

[독립 산술·출처 감사](../research/2026-10-14/lockbox_independent_audit.md)는 별도 파서·라벨 구현·좌표 변환·집합 연산으로 26개 모델을 대조했고, 로컬 전체 아카이브 MD5도 다시 계산했다. 이 감사에는 원래 로컬 출력·학습원 선택 기록이 필요하므로 새 복제본에서는 위 공개 재현 명령을 사용한다. 두 검증의 범위는 다르다.

A2는 이제 관측한 자료다. 이 결과를 보고 바꾼 방법에 A2를 다시 미노출 검증 자료로 사용할 수 없다. 원래 14개 모델 계획과 나중에 따로 고정한 12개 모델 계획은 수정하지 않았으며, 두 학습 예산의 점수를 합쳐 순위를 매기지 않는다. 아래 절차는 완료된 원시 처리 경로를 새 환경에서 재현할 때 사용한다.

## A2 원시 파형부터 전체 재현

모델과 관측 규칙은 A2 파형을 열기 전에 고정했다. 원래 목록은 `research/2026-10-13/lockbox_frozen_manifest_portable.json`이다. 이는 공개 사전등록이 아니며, 과거의 전체 연구나 모든 외부 미노출 이력을 소급해서 보장하지 않는다.

```bash
bash scripts/run_mac.sh download --record 17782542 \
  --file 2024-04-19_TypeA_sample2.7z --max-gib 14 --connections 4 --retries 3
```

다운로드는 이어받을 수 있다. 파일의 논리적 크기가 14.8 GB인 `.part`는 완료 증거가 아니다. 모든 청크와 전체 MD5 검사가 끝나야 한다. 이 파일은 하나의 solid 블록으로 압축되어 있다. 전체 압축 해제량은 약 75.2 GB이므로 속도 폴더별 추출·캐시 후 임시 CSV를 지우는 기존 collection 경로를 사용한다.

공식 7-Zip 실행 파일 `7zz` 또는 `7z`가 PATH에 있어야 한다. Mac용 설치 도구를 Linux용 `setup-tools` 명령으로 대신 실행하지 않는다. 이미 확인한 Mac 바이너리를 `.local-tools/7zip-26.03/`에 두면 `run_mac.sh`가 찾는다.

모델 체크포인트는 Git에 포함하지 않는다. 위 복원 명령을 아직 실행하지 않았다면 먼저 실행한다.

```bash
bash scripts/run_mac.sh python scripts/restore_frozen_models.py
```

다운로드가 진행 중인 별도 터미널이 있을 때 다음 명령은 최대 6시간 기다린 뒤 **A2만** 처리한다. 완료된 아카이브가 이미 있으면 기다리지 않는다. 다운로드를 시작하는 명령은 아니므로, 다운로드 프로세스가 없고 완성 파일도 없으면 실패한다.

```bash
bash scripts/run_mac.sh python scripts/continue_lockbox.py --wait-seconds 21600
```

기존 작업은 완료됐으므로 재현을 위해 중복 프로세스를 띄울 필요가 없다. 새로운 원시 재현 실행의 진행 상태와 로그는 다음 경로로 확인한다. 로그 파일은 실행 시 해당 경로에 출력을 저장한 경우에 존재한다.

```bash
cat reports/research_20261013/lockbox_A2_pipeline_state.json
tail -n 20 reports/research_20261013/holdout_download.log
tail -n 20 reports/research_20261013/lockbox_pipeline.log
```

고정 파일 세 개는 모델 목록 `lockbox_frozen_manifest_portable.json`, 판정·평가 정의 `lockbox_evaluation_spec.json`, 전처리 코드 `lockbox_preprocessing_spec.json`이다. 모두 `research/2026-10-13/`에 있다. 기존 고정을 지우거나 덮어써서 바뀐 코드를 통과시키지 않는다. 기본 출력은 `reports/research_20261013/lockbox_A2/`, 전처리용 별도 경로는 `lockbox_A2_preprocess/`다. 완료된 결과는 덮어쓰지 않는다. 다른 경로를 지정하면 새로운 실행이며 이전 결과도 보존한다.

후속 흐름 후보는 원래 14개 모델을 바꾸지 않고, `transport_lockbox_spec.json`에 별도로 고정했다. 아래 명령은 원래 평가의 성공을 기다린 후 같은 검증된 입력으로 12개 추가 모델을 평가한다. 계수가 명세에 들어 있으므로 과거 학습 보고서나 재학습이 필요 없다. 이 평가도 완료됐으며, 아래 명령은 새 환경에서 원시 경로를 재현할 때 사용한다.

```bash
bash scripts/run_mac.sh python scripts/evaluate_transport_lockbox.py --wait-seconds 21600
```

성공 출력은 `reports/research_20261013/transport_lockbox_A2/`에 저장한다. 보존된 실행 로그는 `reports/research_20261013/transport_lockbox_pipeline.log`다. 주요 진행 상태는 원래 파이프라인 상태 파일로 확인한다. 후보의 주 비교는 0.05/0.10 m/s로 학습한 모델이고, 나머지 두 학습 속도 쌍도 빠짐없이 보조 결과로 저장했다. 이 결과를 원래 단일 속도 학습 모델과 같은 학습 예산의 비교인 것처럼 합치지 않는다.

전체 아카이브 MD5 검사 → 전처리 출처 고정 → 폴더별 압축 해제·감사·캐시 → 고정 모델 평가 순서다. 중간에 실패하면 단계와 원인이 상태 파일에 남는다. collection에서 생성하는 대상별 임시 라벨을 그대로 성능 기준으로 쓰지 않는다. 고정 평가기는 학습원 A의 관측 설정으로 다시 판정한다. 전체 아카이브 검증, 입력 CSV·모델·코드 해시와 표본 ID 확인을 통과한 뒤 결과를 저장한다. 새 결과에 맞춰 모델을 수정하면 그 결과는 다음 방법의 개발 자료가 된다.

노트북이 잠자기에 들어가거나 연결이 끊기면 진행이 멈출 수 있다. 내려받은 청크는 재사용한다. `.download.lock`은 실제 다운로드 프로세스가 살아 있는 동안 수동으로 지우지 않는다. 전처리는 약 25 GB짜리 속도 폴더 하나씩 처리한 뒤 자체 임시 CSV만 지우고, 원시 아카이브는 보존한다. 약 75.2 GB 전체를 동시에 풀지 않는다.
