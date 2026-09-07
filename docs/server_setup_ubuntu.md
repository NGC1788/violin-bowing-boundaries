# Ubuntu 서버 사용 안내

작성 기준: 2026-09-07. 모든 명령은 RustDesk로 접속한 **Ubuntu 서버의 터미널**에서 실행합니다. 서버에는 직접 접속하지 않은 상태에서 만든 안내이며, 실제 환경 점검 결과를 보고 확정합니다.

## 1. 서버 상태 확인

```bash
cat /etc/os-release
uname -m
getconf GNU_LIBC_VERSION
nvidia-smi
free -h
df -h "$HOME"
command -v python3
command -v git
command -v curl
```

- 아키텍처 `x86_64`, glibc 2.28 이상이 필요합니다. Ubuntu 20.04 이상을 대상으로 합니다.
- `nvidia-smi`에 실제 GPU 모델과 드라이버가 나와야 합니다. 기존 실행 중인 작업이 있으면 그 작업을 종료하지 말고 점유 상태를 확인합니다.
- 설치 사전 점검은 작업 폴더에 **최소 20 GiB 여유 공간**을 요구합니다. 이는 설치 캐시 등을 위한 운영 기준이며, 모든 서버에서 충분하다는 보장은 아닙니다. 30 GiB 이상을 권하며 원시자료·압축 해제 공간은 별도입니다. 설치 후의 작은 실행 점검은 1 GiB 기준을 적용합니다. bootstrap을 다시 실행하거나 환경을 갱신할 때에는 사전 설치 기준을 다시 확인합니다.
- CUDA 12.x 최소 minor compatibility 드라이버는 Linux 525.60.13입니다. 560.35.05 미만에서는 일부 JIT/PTX 기능의 제한이 있으므로 우선 eager 실행을 점검하고 초기에는 `torch.compile`을 쓰지 않습니다. 그보다 낮거나 GPU 조회가 실패하면 드라이버를 임의로 교체하지 말고 서버 관리자에게 현 상태 점검을 요청합니다. [NVIDIA 공식 버전 표](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html).
- `nvidia-smi`의 `CUDA Version` 표시는 이 프로젝트에 설치된 PyTorch 런타임 버전과 다른 값일 수 있습니다.

`git`, `curl`, `python3`가 없을 때만 다음을 사용합니다. 학교 계정에 sudo 권한이 없으면 관리자에게 필요한 패키지만 요청합니다.

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends git curl ca-certificates python3
```

영상 조사와 장시간 작업에는 다음 도구도 유용합니다. 초기 GPU 설치에 필수는 아닙니다.

```bash
sudo apt-get install --no-install-recommends ffmpeg tmux p7zip-full
```

## 2. GitHub에서 받기

```bash
mkdir -p "$HOME/research"
cd "$HOME/research"
git clone https://github.com/NGC1788/violin-bowing-boundaries.git
cd violin-bowing-boundaries
```

공개 저장소이므로 서버에 GitHub 로그인이나 토큰을 넣을 필요가 없습니다. 이미 clone한 경우 다시 clone하지 말고 기존 폴더로 들어갑니다. 홈 디스크가 작다면 첫 clone부터 여유 공간이 있는 **사용자에게 쓰기 권한이 있는** 다른 경로를 사용합니다. 서버의 공유 폴더 권한을 바꾸지 않습니다.

## 3. 사전 점검

```bash
python3 scripts/doctor.py --preflight --require-cuda
```

이 단계는 PyTorch를 설치하거나 실행하지 않습니다. OS·glibc·GPU 드라이버·RAM·디스크와 도구 존재 여부를 확인합니다. 실패 시 오류를 먼저 해결합니다. 보고서는 `reports/environment/`에 저장됩니다.

## 4. 설치와 실제 실행 확인

```bash
bash scripts/bootstrap_ubuntu.sh
```

설치 스크립트는 다음을 수행합니다.

1. 사전 점검을 다시 수행합니다.
2. `uv`가 없으면 공식 설치기로 사용자 경로에 설치합니다. 쉘 설정 파일은 변경하지 않습니다.
3. 프로젝트 `.venv`에 고정된 Python/패키지를 설치합니다. 캐시와 관리 Python은 프로젝트 `.cache/`에 둡니다.
4. 실제 작은 GPU 연산과 역전파, GPyTorch 커널 계산을 확인합니다.
5. 합성 WAV의 24-bit 왕복, 리샘플링, 멜 스펙트럼, 이미지·테이블 저장, 그림 생성을 확인합니다.
6. 환경 보고서와 패키지 목록을 저장합니다.

최종 `SETUP PASS`가 나와야 이 설치 순서가 완료된 것입니다. 설치 시간은 다운로드 속도에 따라 달라집니다. 다운로드 도중 끊겼으면 같은 설치 명령을 다시 실행하면 됩니다. 오류가 같은 위치에서 반복되면 출력 마지막 부분을 확인하고 원인을 해결합니다.

시스템 NVIDIA 드라이버나 `/usr/local/cuda`를 설치·삭제하지 않습니다. PyTorch 실행에 필요한 CUDA 라이브러리는 프로젝트 의존성으로 설치됩니다. 이 구성에서는 별도 시스템 CUDA Toolkit을 먼저 설치할 필요가 없습니다.

설치 뒤 다시 확인할 때:

```bash
bash scripts/run.sh doctor
bash scripts/run.sh smoke
```

별도 `conda activate`, 전역 `pip install`, 가상환경 활성화를 할 필요가 없습니다. 의존성을 추가할 때에는 `pyproject.toml`과 `uv.lock`을 함께 수정·검토합니다. 논문 코드의 오래된 `requirements.txt`를 이 환경에 그대로 덮어 설치하지 않습니다.

## 5. 공개 데이터 목록만 확인

```bash
bash scripts/run.sh catalog
```

기본값은 Parts 1, 2, 6입니다. 요청한 레코드와 실제 반환된 버전 ID, 파일명, 압축 용량, MD5, 조회 시각, 원본 API 응답을 저장합니다.

2026-09-07 공식 API에서 확인한 규모:

| Part | 요청 ID → 실제 응답 ID | 압축 총량 |
|---|---|---:|
| 1 | 17749110 → 17749111 | 40.972 GiB, 파일 3개 |
| 2 | 17782542 → 17782542 | 40.598 GiB, 파일 3개 |
| 6 | 17795326 → 17795326 | 26.281 GiB, 파일 1개 |

Part 1·2도 아카이브 하나가 약 13–14 GiB입니다. 처음부터 모두 내려받지 않습니다. [Part 1](https://zenodo.org/records/17749111), [Part 2](https://zenodo.org/records/17782542), [Part 6](https://zenodo.org/records/17795326).

**오늘 기본 설치의 완료 지점은 여기까지입니다.** 다음 데이터 선택은 가용 디스크, 현·속도·세션 구성, 필요한 상태 판정 구간을 확인한 뒤 합니다.

명시적으로 파일을 선택한 이후의 다운로드 예시입니다. 아래는 Part 1의 압축파일 **13.116 GiB** 한 개를 받는 명령입니다. 목록에서 같은 파일의 존재와 실제 크기를 확인하고 사용합니다.

```bash
bash scripts/run.sh download \
  --record 17749111 \
  --file '2024-03-25_TypeA_sample1.7z' \
  --max-gib 14
```

`--max-gib`는 **선택한 압축파일 한 개의 상한**입니다. 총 데이터 용량이나 압축 해제 용량 제한이 아닙니다. 자동 압축 해제는 하지 않습니다. 완료 후 MD5가 맞는 파일만 최종 파일명으로 게시합니다. 끊긴 다운로드는 동일 명령으로 재개합니다. 체크섬이 틀리면 파일을 그대로 사용하지 않습니다.

압축 해제 전에 목록과 총 비압축 크기를 확인합니다. 다운로드 결과에 표시된 실제 경로를 사용합니다.

```bash
7z l data/raw/zenodo/17749111/2024-03-25_TypeA_sample1.7z
df -h .
```

`7z` 대신 `7zz`만 설치된 환경은 해당 명령을 사용합니다. 이후 선택한 파일을 `data/interim/`의 별도 폴더로 해제합니다. 모든 큰 CSV를 한 번에 RAM으로 읽는 분석부터 시작하지 않습니다.

## 6. Jupyter 사용 — 선택 사항

```bash
bash scripts/run.sh jupyter
```

출력된 `http://127.0.0.1:8888/...token=...` 주소를 **Ubuntu 안의 브라우저**에 붙여넣습니다. Mac 브라우저에 입력하면 Mac 자신을 가리킵니다. RustDesk에서는 Ubuntu 브라우저 화면을 보면 됩니다. 토큰을 채팅이나 GitHub에 올리지 않습니다. 종료는 실행한 터미널에서 `Ctrl+C`입니다.

## 7. 작업을 오래 실행할 때 — 선택 사항

```bash
tmux new -s violin
```

그 터미널에서 작업을 실행합니다. `Ctrl+B`를 누른 뒤 손을 떼고 `D`를 누르면 세션을 남기고 나옵니다. 다시 들어가려면:

```bash
tmux attach -t violin
```

RustDesk 연결 종료와 실제 Ubuntu 로그아웃·재부팅은 다릅니다. tmux도 서버 재부팅 뒤 작업을 복구하지는 않으므로 장시간 학습에는 체크포인트 저장을 구현해야 합니다.

## 8. 다음부터 코드 업데이트

학습이나 다운로드가 돌고 있지 않은 시점에 저장소 폴더에서:

```bash
git status --short
git pull --ff-only
bash scripts/bootstrap_ubuntu.sh
```

직접 수정한 코드 때문에 pull이 막히면 변경을 보존하고 확인합니다. `git reset --hard`, `git clean`으로 정리하지 않습니다. 실행 결과와 원시자료는 기본 `.gitignore` 대상이어서 일반적인 업데이트로 교체되지 않습니다.

## 9. 문제별 대응

| 증상 | 다음 조치 |
|---|---|
| `nvidia-smi` 없음/실패 | OS 설치보다 GPU/드라이버 상태 확인이 먼저입니다. 관리자에게 출력 전달 |
| `Permission denied` | 본인에게 쓰기 권한이 있는 폴더로 clone. 전체 설치를 sudo로 실행하지 않기 |
| `No space left on device` | `df -h .` 확인. 다른 사람의 파일·캐시를 삭제하지 않기 |
| CUDA unavailable | doctor 보고서의 torch build, driver, visible GPU 오류 확인 |
| CUDA out of memory | 점유 상태 확인 후 작업 시간 조정. 다른 작업을 강제 종료하지 않기 |
| `uv: command not found` | 직접 uv 대신 `bash scripts/run.sh ...` 사용. 없으면 bootstrap 재실행 |
| 공유 라이브러리/import 오류 | 정확한 오류와 Python/패키지 버전 기록. 무작위 downgrade 하지 않기 |
| Zenodo timeout/429 | 잠시 후 catalog 또는 같은 다운로드 명령 재시도. 반복 요청 병렬화하지 않기 |
| download lock 존재 | 같은 파일 다운로드가 실행 중인지 확인. 실행 중이 아니라고 확인된 오래된 lock만 제거 |

## 10. 설치 후 공유할 정보

다음 두 명령의 출력으로 GPU 실행 여부와 파일 목록을 확인할 수 있습니다.

```bash
bash scripts/run.sh doctor
bash scripts/run.sh catalog
```

`SETUP PASS` 여부, GPU/드라이버·남은 디스크, 오류가 있다면 마지막 부분을 연구 기록에 남깁니다. Jupyter 접속 토큰은 제외합니다. 논문에 사용할 각 실행은 코드 commit, 설정, 데이터 버전·체크섬, split, seed, 출력 경로를 함께 기록해야 합니다.
