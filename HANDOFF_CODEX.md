# 인수인계 — 활-현 연주 가능 경계 연구 (Claude → Codex)

작성 2026-10-13 · 대상: 이 저장소를 이어받아 작업할 에이전트
저장소 `NGC1788/violin-bowing-boundaries` · 로컬 `~/violin-bowing-boundaries` · main `f380e9f` (origin과 동일, 미푸시 없음)

---

## 0. 30초 요약

로봇이 켠 첼로 공개 데이터 24,000 strokes에서 **Helmholtz motion이 나는 (활 위치 β, 활 압력) 영역**을 측정하고, 물리 모델(digital waveguide)이 **보정에 쓰지 않은 조건에서도** 그 영역을 맞히는지를 평가한다. 최종 목표는 ① 물리 모델 vs 데이터 기반 모델을 같은 데이터·같은 분류기·같은 지표로 비교, ② 첼로에서 세운 절차를 바이올린으로 이식.

중간발표(2026-09-22)까지 ①의 물리 모델 쪽이 끝났다. 데이터 기반 모델은 미착수. **계산 서버가 2026-09-23 포맷되어 현재 GPU 없음**, 원본 데이터도 서버와 함께 사라졌다(재다운로드 가능, 분석 결과는 전부 백업됨).

---

## 1. 지금 당장 알아야 할 제약

| 항목 | 상태 |
|---|---|
| GPU 서버 | tako-server(A5000 24GB, `tako@100.86.117.70`)가 9/23 클러스터 재구성으로 포맷. 10/13 현재 SSH 응답 없음 |
| 원본 데이터 | `data/` **전무**. Zenodo 재다운로드 필요(아래 3절) |
| 분석 결과 | `reports/` 111 MB 전부 보존 — 라벨·경계·보정 흔적·시뮬 결과 |
| 코드 | 전부 push 완료. 테스트 13개 파일 모두 통과 상태로 남김 |
| 최종 발표일 | **미확인.** 일정 산정의 전제가 비어 있음 |

맥에서 git 명령은 Xcode 라이선스 때문에 막혀 있다. push가 필요하면 `gh api` git-data API를 쓴다. **force push 금지**(사용자 지시).

---

## 2. 저장소 구조와 실행법

```
scripts/run.sh <command>     # 모든 진입점. 내부에서 uv run --locked
```

| 명령 | 스크립트 | 하는 일 |
|---|---|---|
| `doctor` | doctor.py | 환경·CUDA 점검 |
| `setup-tools` | install_7zip.py | sudo 없이 7-Zip 설치(학교 계정용) |
| `smoke` | smoke_test.py | 합성 신호로 파이프라인 점검 |
| `catalog` / `download` | zenodo_catalog.py | Zenodo 메타데이터 조회 / MD5 검증 병렬 다운로드 |
| `archive-audit` | archive_audit.py | 7z 목록·크기 검사 |
| `trial-audit` | trial_audit.py | 시행별 구조 감사 + 창 캐시 생성 |
| `grid-summary` | grid_summary.py | 격자 요약 |
| `regime-map` | regime_map.py | **분류기 본체** + 경계 적합 + 그림 |
| `collection` | collection.py | 아카이브 17개 전체 처리 오케스트레이션 |
| `simulate` | simulate_grid.py | 물리 모델로 같은 격자 재현 |
| `calibrate` | calibrate.py | 마찰·손실 5개 파라미터 탐색 |

문서는 `docs/`에 있다: `data_contract.md`(데이터 규약), `regime_map.md`(분류기), `collection.md`, `simulation.md`, `validation.md`(**모든 결함과 수정의 감사 추적 — 제일 먼저 읽을 것**), `research_roadmap.md`, `final_violin_check.md`.

환경: uv + `uv.lock`, Python 3.11. GPU 경로는 torch, CPU 경로는 numpy.

⚠️ **`uv.lock`은 Linux x86_64 전용으로 고정되어 있다**(`pyproject.toml`의 `[tool.uv] environments`, CUDA 휠 때문). 맥에서 `scripts/run.sh`나 `uv run --locked`를 실행하면 `The current Python platform is not compatible with the lockfile's supported environments`로 실패한다. 맥에서 분석 코드를 돌리려면 별도 가상환경을 만들어 스크립트를 직접 호출한다:

```bash
uv venv --python 3.12 .venv-mac
uv pip install --python .venv-mac numpy matplotlib torch    # 분류·그림·시뮬레이션에 이것만 필요
PYTHONPATH=scripts .venv-mac/bin/python scripts/regime_map.py --help
```

`scripts/*.py`가 실제로 import하는 외부 패키지는 **numpy 하나**이고, 그림에 matplotlib, 시뮬레이션·보정에 torch가 추가로 필요하다. pandas·scipy·librosa 등은 `pyproject.toml`에 선언되어 있을 뿐 분석 경로에서 쓰이지 않는다.

---

## 3. 데이터 복구 절차

공개 데이터는 Zenodo 로봇 운궁 첼로 데이터셋(IWK, 빈 음대). 아카이브 17개가 `collection.py`의 `QUEUE`에 처리 순서대로 박혀 있다.

```bash
bash scripts/run.sh catalog                       # 메타데이터만
bash scripts/run.sh collection --limit 1          # 첫 아카이브부터 재개 가능
```

- 전체는 수십 GB. **T1(기준선)을 다시 돌리려면 A현 아카이브 `2024-03-25_TypeA_sample1.7z` 하나면 된다.**
- 다운로드는 64 MiB 청크 병렬(`PARALLEL_CHUNK`), 206/Content-Range 정확히 검사, `.part.chunks.json`으로 재개. MD5는 전체 파일 기준.
- 처리는 **속도 폴더 단위**: 폴더 하나 추출 → 감사 → 창 캐시(`window_N.npy` float32 + `profile_N.npy` 1 kHz, `PROFILE_DECIMATION=50`) 저장 → **추출한 CSV 삭제**. 삭제 규칙은 `collection.py` 모듈 docstring에 명문화되어 있다. 기존 추출본(`data/interim/typeA_sample1`)은 읽기만 하고 수정하지 않는다.

---

## 4. 분류기 (regime_map.py) — 이 연구의 심장

**입력은 bridge force 파형 하나뿐이다. 활 압력·속도·β는 분류에 절대 쓰지 않는다.** 이걸 어기면 "경계를 측정했다"는 주장 자체가 무너진다.

절차: 0.4 s 분석창 → 주기 추정 → **모든 주기를 접어 평균 파형** → 히스테리시스(3 표준오차) 전환점 → 가장 큰 flyback의 `flyback_frac`(0.5) 이상인 flyback 개수 → 라벨.

```python
DEFAULT_THRESHOLDS = {"flyback_frac": 0.5, "hysteresis_sigma": 3.0, "noise_floor_factor": 3.0,
                      "min_amplitude": 0.02, "min_periodicity": 0.8, "max_aperiodic": 0.5,
                      "subharmonic_ratio": 0.75, "max_hole": 2}
REFERENCE_IMPEDANCE = 1.175
SENSITIVITY_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7)
```

라벨: `helmholtz` / `multiple_slip` / `subharmonic` / `aperiodic` / `no_oscillation` / `ambiguous`.

### 반드시 유지해야 하는 두 가지

1. **잡음 바닥 판정은 AND다.**
   ```python
   if not (amplitude >= thresholds["min_amplitude"]) and not (periodicity >= thresholds["min_periodicity"]):
       return "no_oscillation"
   ```
   OR로 바꾸면 β가 큰 영역의 약한 주기 운동이 통째로 회색이 된다(한 번 겪은 버그).
2. **주기 접기 전에 세지 말 것.** v1은 샘플 단위 |diff| 임계로 셌고, 깨끗한 톱니파에서 flyback 수가 0.03~2.83으로 흔들렸다. 접기 도입 후 애매 판정 1,300 → 20.

### 경계 적합 (`fit_boundaries`)
β 레벨마다 Helmholtz 런을 찾고(`max_hole=2`까지 구멍 메움), 런 양옆 힘의 기하평균을 경계로 잡는다. 런이 힘 격자 끝에 닿으면 **censored** 처리하고 적합에서 뺀다. `excluded` 셀(속도 평탄 구간을 벗어난 창)은 Helmholtz로 치지 않고, 그 옆 경계도 censored.

---

## 5. 라벨 검증 — flyback 법칙

이상적 Helmholtz motion이면 bridge force의 급격한 변화량이 `ΔF = 2 Z v_b / β`. 분류기는 이 식도, v_b도 β도 모른다. 따라서 일치하면 **독립 근거**가 된다.

| 현 | n | 속도 지수 | β 지수 | Z (median, kg/s) |
|---|---|---|---|---|
| A | 2,190 | +1.015 ± 0.010 | −0.992 ± 0.008 | 0.9463 |
| B | 3,086 | +1.016 ± 0.007 | −0.908 ± 0.006 | 0.9450 |
| C | 3,229 | +0.984 ± 0.006 | −0.866 ± 0.005 | 0.9563 |
| D | 3,171 | +0.992 ± 0.005 | −0.914 ± 0.005 | 0.9584 |

이론값 +1 / −1. **한계: 이건 "물리 법칙과 모순되지 않는다"이지 "라벨이 전부 옳다"가 아니다.** 원저자 시행별 기준 라벨은 공개되지 않았다(→ T7).

측정 경계 기울기(격자 집계, 이론 하한 −2 / 상한 −1):

| 현 | lower | upper |
|---|---|---|
| A | −1.52 ± 0.06 (n=107) | −0.69 ± 0.33 (n=34) |
| B | −1.72 ± 0.07 (n=76) | −0.97 ± 0.10 (n=38) |
| C | −1.50 ± 0.07 (n=79) | −1.08 ± 0.12 (n=48) |
| D | −1.66 ± 0.05 (n=69) | −0.57 ± 0.11 (n=46) |

---

## 6. 물리 모델 (bowed_string.py)

MSW 계열 **digital waveguide**. 활을 기준으로 bridge 쪽/nut 쪽 두 지연선, 왕복마다 부호 반전 + 1차 저역 반사 필터(DC 이득이 Q1을, 차단주파수가 코너 둥글기를 정함, 군지연은 지연선에서 뺌). 활 접점에서 현은 정확히 2Z를 본다. 마찰은 `μ(w) = μ_d + (μ_s − μ_d)·v0/(v0+|w|)`, stick/slip 히스테리시스 포함 2차식 닫힌 해. bridge force = 2Z × 도착 파.

```python
StringParams(f0=98.08, impedance=0.946, length=0.7, q1=2000, corner_hz=6000)
FrictionParams(mu_s=0.6725, mu_d=0.346, v0=0.06)   # 문헌값(Mores DAGA 2016)
```

**모달(stiff-string + point bow) 버전은 폐기했다.** 점 활에서 한 스텝 bow admittance가 시간 간격에 의존해(50 kHz 2.0 kg/s, 100 kHz 2.7 vs Z=0.946) 마찰 해가 수렴하지 않는다. 되살리지 말 것. 이유는 `docs/simulation.md`와 `validation.md`에 기록되어 있다.

**GPU 함정(한 번 터짐):** per-stroke 마찰 값을 numpy로 만들면 CUDA 텐서와 만나 `can't convert cuda:0 device type tensor to numpy`가 난다. `simulate()` 안에서 반드시 백엔드로 올린다 — `FrictionParams(*(xp.asarray(spread(v)) for v in (...)))`. numpy↔torch 동치 테스트가 `tests/test_bowed_string.py`에 있으니 지우지 말 것.

구동은 **실제 로봇이 기록한 활 속도·압력 시계열**을 그대로 넣는다(합성 입력 아님). 시뮬레이션 레이트 100 kHz, 측정과 같은 관측 잡음(A현 0.068045)을 더하고 **같은 분류기**로 라벨링해 비교한다.

---

## 7. 보정 설계와 결과 (calibrate.py)

```python
SPACE = {"mu_s": (0.45, 1.20, False), "mu_d": (0.15, 0.55, False), "v0": (0.01, 0.40, True),
         "q1": (300.0, 6000.0, True), "corner_hz": (1500.0, 20000.0, True)}   # (하한, 상한, 로그스케일)
```

**평가 설계가 이 연구의 핵심 주장이다.** 12칸(현 4 × 속도 3) 중 **A현 0.10 m/s 한 칸에서만** 5개 파라미터를 맞추고, 나머지 11칸 18,000 strokes는 손대지 않는다. 지표는 Helmholtz 영역의 IoU.

| | 보정 칸 (500셀) | hold-out 속도 (1,000셀) |
|---|---|---|
| 문헌값 | IoU 0.3521 / 일치 0.594 | IoU 0.4565 / 일치 0.636 |
| 최적값 | **IoU 0.7382** / 일치 0.710 | **IoU 0.6611** / 일치 0.703 |

최적 파라미터 (`reports/calibration/2024-03-25_TypeA_sample1_rate50000.jsonl`, index 550, 총 1,500벌):
```json
{"mu_s": 0.91246, "mu_d": 0.26251, "v0": 0.06066, "q1": 3726.62, "corner_hz": 1896.41}
```
⚠️ `corner_hz` 1896이 탐색 하한 1500에 가깝다. **더 좋은 값이 상자 밖에 있을 수 있다 → T2.**

구현 메모: 파라미터 세트를 16벌씩 **한 배치에 묶어** 평가한다(스텝 비용이 커널 런치에 지배되므로 16벌이 1벌과 비슷한 비용). 평가 1벌 ≈ 29초. 결과는 JSONL append라 중단·재개·감사가 된다. 재개 시 평가 0번(문헌값)이 난수를 소비하지 않아 정렬이 어긋났던 버그를 고쳤으니, `draw`는 항상 호출해야 한다.

전체 격자(6,000 strokes) 재현 결과:

| 격자 | 파라미터 | IoU | 일치율 | 비고 |
|---|---|---|---|---|
| A | 문헌값 | 0.4391 | 0.627 | **기준선(run `20260919T081019Z_ec4f505b`)** |
| A | 보정값 | 0.5969 | 0.670 | 보정에 쓴 현 |
| B | 보정값 | 0.6337 | 0.678 | 미사용 현 |
| C | 보정값 | 0.7347 | 0.782 | 미사용 현 |
| D | 보정값 | 0.7361 | 0.780 | 미사용 현 |

A현 1격자 ≈ 22분(torch, A5000).

**재현 실패 영역:** 아래 경계는 위치·기울기까지 맞지만 **위 경계(큰 힘 쪽)는 실측과 크게 다르다**(D현 실측 upper −1.09±0.06 vs 모델 −2.67±0.28). 점 활 근사의 한계로 본다 → T3.

---

## 8. 남은 과제

| | 과제 | 선행 | 환경 |
|---|---|---|---|
| T1 | 문헌값 기준선 — **A현은 끝남(0.4391)**, B·C·D 3격자 남음 | 데이터 | GPU |
| T2 | 탐색 범위 확대 재보정 (`corner_hz` 하한 문제) | 데이터 | GPU |
| T3 | upper boundary 재현 — 활털 유한 폭, 현 비틀림, 송진 열 마찰 | T2 | 모델링 |
| T4 | data-driven model 구현 (PINN / neural operator) + **동일 평가 설계** | — | 설계는 CPU, 학습 GPU |
| T5 | 두 모델 비교표 — 이 연구의 결론 | T2, T4 | — |
| T6 | 바이올린 적용 — 녹음 프로토콜부터 | — | 불필요 |
| T7 | 분류 신뢰도 보강 — 표본 육안 검증 또는 원저자 라벨 요청 | — | 불필요 |
| T8 | 최종 보고서·발표 | 전부 | — |

T4 참고 문헌: Luan & Scavone, arXiv:2505.18950 (bow-string 마찰의 PINN/PI-DeepONet). 데이터셋·모델 쪽은 Pàmies-Vilà et al. *Acta Acustica* 8, 45 (2024), Lampis et al. 8, 44 (2024), van Walstijn et al. 10, 47 (2026).

---

## 9. 파일 위치

| 무엇 | 경로 |
|---|---|
| 코드·결과 | `~/violin-bowing-boundaries` (reports/ 포함) |
| 서버 백업분 | `~/violin-bowing-boundaries/_server_extra/` (midterm_results.tar.gz, 서버 환경 스냅샷) |
| 중간발표 pptx·PDF·대본 | `~/Downloads/KSHS-KERAS 김태건 정지윤 정지훈*.pdf`, `R&E_중간발표_*` |
| 발표 그림 생성 스크립트·자산 | `~/Desktop/handoff_20260919/v3/` (mkfigs.py, mkdiag.py, build3.py, fig/, assets/) |
| 논문 그림 143장 + 전사·조사 자료 | `~/Desktop/handoff_20260919/` |
| 남은 과제 요약 페이지 | https://claude.ai/artifact/UWE2o1nBciuSK15NjUPYmy |

---

## 10. 작업 규칙 (사용자 지시 — 반드시 지킬 것)

- **추가 구매 금지.** 센서 제작·대규모 데이터 수집 없음.
- **force push 금지.** 맥 git은 Xcode 라이선스로 막혀 있어 `gh api` 사용.
- 원시 데이터·녹음·자격증명을 공개 저장소에 올리지 않는다(`data/`, `reports/`, `logs/`, `.venv/`, `.cache/`는 gitignore).
- 서버는 빌린 것이고 sudo가 없다. 패키지 설치는 프로젝트 안에서(`setup-tools`), 다른 사용자 자료는 건드리지 않는다.
- 숫자는 전부 재현 가능한 산출물에서 가져온다. 추정값을 단정적으로 쓰지 않는다.
- 결함을 고치면 `docs/validation.md`에 증상·원인·수정·재발 방지 테스트를 남긴다. 이 파일이 이 연구의 신뢰도 근거다.

---

## 11. 인수 직후 점검 순서

**맥에서 (GPU 없이, 산출물 무결성만 확인)** — 의존성 없이 바로 된다:

```bash
cd ~/violin-bowing-boundaries
python3 -c "import json;d=json.load(open('reports/simulation/2024-04-15_TypeD_sample1__2024-04-17/20260919T074717Z_f3212a15/comparison.json'));print(d['comparison']['helmholtz_iou'])"
# → 0.7361 이 나오면 결과 산출물이 온전한 것
git log --oneline -1    # f380e9f
```

맥에서 코드까지 돌려보려면 2절의 `.venv-mac` 레시피로 환경을 만든 뒤 `PYTHONPATH=scripts .venv-mac/bin/python -m pytest tests -q`.

**Linux GPU 서버에서 (복구 후)**:

```bash
bash scripts/run.sh doctor          # CUDA 포함 환경 점검
uv run --locked pytest -q           # 테스트 13개 파일
bash scripts/run.sh smoke           # 합성 신호 파이프라인 점검
```

그다음 `docs/validation.md` → `docs/regime_map.md` → `docs/simulation.md` 순으로 읽으면 전체 맥락이 복원된다.
