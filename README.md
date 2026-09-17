# Violin Bowing Boundaries

**바이올린 활–현 연주 가능 경계의 측정과 예측 — 물리 모델과 데이터 기반 모델의 비교.**

불확실성을 고려한 활–현 경계 예측 연구를 위한 저장소입니다. 현재 버전은 Ubuntu GPU 환경 설치, 환경·합성 신호 점검, 공개 데이터 다운로드·자동 재개, 압축 목록 검사와 제한된 추출·CSV 앞부분 확인, 전체 시행의 구조 감사를 제공합니다. **학습 모델과 실제 연구 결과는 아직 구현 또는 검증되지 않았습니다.**

**2026-09-13 실행안:** 추가 구매 없이 서버에서 방법 개발과 본 검증을 수행합니다. 자체 힘 센서 제작과 대규모 학습 자료 수집은 제외하고, 마지막에 보유 바이올린과 녹음 장비로 제한적인 적용 확인을 합니다. [수정한 연구 계획](docs/research_roadmap.md), [마지막 바이올린 확인의 범위](docs/final_violin_check.md).

## 서버에서 시작

**이미 설치를 완료했다면 [오늘 서버에서 할 일](docs/today_server.md)부터 진행합니다.** 기존 다운로드를 이어받아 첫 데이터 파일을 준비하는 순서입니다. 첫 데이터 준비가 `PASS`라면 다음은 [전체 시행 감사](docs/trial_audit.md)입니다.

**sudo가 없는 학교 계정도 진행할 수 있습니다.** `bash scripts/run.sh setup-tools`로 공식 7-Zip을 프로젝트 안에 설치하고, 안내의 `nohup` 명령으로 작업을 유지합니다. tmux 설치는 필요하지 않습니다.

RustDesk로 Ubuntu에 접속한 뒤 **Ubuntu의 터미널**에서 실행합니다. NVIDIA GPU가 있는 Linux x86_64용이며, Mac에서 이 환경을 설치하지 않습니다.

먼저 기존 상태를 확인합니다.

```bash
cat /etc/os-release
uname -m
nvidia-smi
free -h
df -h "$HOME"
```

연구 폴더를 받을 위치는 여유 공간이 있는 사용자 소유 경로로 정합니다. 아래는 홈 폴더 예시입니다.

```bash
mkdir -p "$HOME/research"
cd "$HOME/research"
git clone https://github.com/NGC1788/violin-bowing-boundaries.git
cd violin-bowing-boundaries
```

설치 전 점검과 설치를 차례로 실행합니다. 한 단계가 실패하면 다음 단계로 진행하지 말고 오류를 확인합니다.

```bash
python3 scripts/doctor.py --preflight --require-cuda
bash scripts/bootstrap_ubuntu.sh
```

`SETUP PASS`가 나오면 공개 데이터 **메타데이터만** 조회합니다. 이 명령은 수십 GB의 원시파일을 받지 않습니다.

```bash
bash scripts/run.sh catalog
```

전체 안내와 오류별 대응: [Ubuntu 서버 사용 안내](docs/server_setup_ubuntu.md).

## 포함된 도구

| 파일 | 역할 |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | Python 및 패키지 버전 기록, Linux CUDA 환경 고정 |
| `scripts/bootstrap_ubuntu.sh` | 사전 점검 → 프로젝트 환경 설치 → GPU/합성 신호 점검 |
| `scripts/doctor.py` | GPU·메모리·디스크·버전 기록, 설치 후 실제 CUDA 순전파·역전파 |
| `scripts/smoke_test.py` | 합성 WAV·리샘플링·멜 스펙트럼·PNG·Parquet·간단한 회귀분류 실행 점검 |
| `scripts/zenodo_catalog.py` | Zenodo 버전·용량·체크섬 보존, 파일 단위 다운로드·재개 |
| `scripts/archive_audit.py` | 압축 해제 없이 멤버 목록·크기·경로·블록 검사 |
| `scripts/prepare_first.py` | 첫 파일 준비, 용량을 제한한 추출, CSV 앞부분 확인과 보고서 출력 |
| `scripts/trial_audit.py` | 추출된 전체 시행의 파일 짝·열 수·유한값·분석 창 범위·채널 범위·속도 평탄구간 감사 |
| `scripts/install_7zip.py` | 관리자 권한 없이 공식 Linux 7-Zip 설치·SHA256 및 실행 검사 |
| `scripts/run.sh` | 매번 환경 활성화 없이 프로젝트 명령 실행 |
| `tests/` | 다운로드 및 환경 점검 도구의 회귀 검증 |

초기 환경은 Python 3.11.15, PyTorch 2.14.0 CUDA 12.6, torchvision 0.29.0, torchaudio 2.11.0을 사용합니다. 그 외 실제 선택된 버전과 배포 파일 해시는 `uv.lock`에 기록합니다. `uv.lock` 생성은 설치 호환성의 모든 측면을 검증하지 않으므로, 서버에서 실제 GPU 실행 점검이 필요합니다.

PyTorch 2.14.0 공식 릴리스와 TorchAudio의 안정 ABI 호환 표를 확인했습니다. 이후 연구를 시작하면 무조건 최신으로 업데이트하지 않고, 사용한 commit과 환경을 고정합니다. [PyTorch 릴리스](https://github.com/pytorch/pytorch/releases/tag/v2.14.0), [TorchAudio 호환 표](https://docs.pytorch.org/audio/stable/installation.html), [uv의 PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 연구 자료와 검증

- [데이터 계약 및 공개 자료 해석](docs/data_contract.md)
- [연구 단계와 검증 기준](docs/research_roadmap.md)
- [이 저장소에서 실제 수행한 검증](docs/validation.md)
- [설치 성공 후 첫 공개 아카이브 확인](docs/first_archive.md)

원본 녹음·영상·센서 파일은 서버의 `data/`에 보존합니다. 이 폴더와 모델 가중치·설치 캐시·실행 로그는 Git 추적 대상에서 제외했습니다. 코드와 문서 수정은 Git으로 공유하고, 연구 데이터 백업은 별도로 관리합니다.

공개 기계 자료의 브리지 힘을 마이크 음압으로 간주하지 않습니다. 공개 자료의 물리 상태와 실악기 녹음의 운영적 안정음 판정은 별도 endpoint입니다. 합성 신호 설치 점검 통과는 실제 바이올린 검출 성능의 증거가 아닙니다.
