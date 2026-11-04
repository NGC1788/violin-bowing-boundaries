"""Write evolve_v4_colab.ipynb (run: .venv-mac/bin/python -m agentic_research.evolve_v4.colab.make_notebook)."""
import json
from pathlib import Path

BRANCH = "research/agentic-forecast-20261030"
REPO = "https://github.com/NGC1788/violin-bowing-boundaries.git"
BUNDLE = "agentic_research/evolve_v4/colab/bundle"

CELLS = [
    ("md", """# evolve_v4 — 코랩 진화 루프 (과제 A: 조회 수 줄이기)

공개 코딩 모델(vLLM, 이 런타임 **안에서만** 실행)이 조회 정책 코드를 계속 새로 쓰고, 고정된 정확 검증기로 채점합니다.

- 번들에는 **search 폴더(A/B)만** 들어 있습니다. 봉인된 C/D는 없으므로 여기서는 holdout을 볼 수 없고, 봉인 평가는 맥에서 1회만 합니다.
- 원장(`ledger.sqlite`)은 런타임 디스크에서 돌고, 5분마다 구글 드라이브 `MyDrive/evolve_v4/<RUN_ID>/`로 안전하게 복사됩니다. 연결이 끊겨도 같은 RUN_ID로 다시 실행하면 이어집니다.
- 계정마다 `RUN_ID`와 `SEED`를 다르게 주세요(예: acct1/1, acct2/2, acct3/3).
- 끝나면 드라이브의 `ledger.sqlite`를 내려받아 맥으로 가져오면 됩니다.

**사용법:** 런타임 → 런타임 유형 변경 → **GPU(G4·H100·A100, 메모리 70GB 이상 권장)** → 아래 설정 칸의 `RUN_ID`·`SEED`만 바꾸고 → **런타임 → 모두 실행**.
- 드라이브 연결 창이 뜨면 허용을 눌러야 다음 칸으로 넘어갑니다.
- 설치 뒤 "세션 다시 시작" 안내가 떠도 **누르지 마세요**(모델은 별도 프로세스라 필요 없음).
- 진화는 백그라운드로 `HOURS` 동안 돕니다. 진행 상황 칸은 아무 때나 다시 실행해서 볼 수 있고, 탭은 가능하면 열어 두세요."""),
    ("code", """# ===== 설정 (계정마다 바꾸기) =====
RUN_ID = "acct1"      # acct1 / acct2 / acct3
SEED = 1              # 1 / 2 / 3
HOURS = 10            # 진화 시간 (Pro+ 최대 연속 실행 24시간)
MODEL = "auto"        # auto = Qwen3.8-27B(2026-08), 안 되면 gpt-oss-120b
EFFORT = "medium"     # gpt-oss 추론 강도: medium 권장 (high는 생각만 하다 답이 잘림)"""),
    ("code", """import subprocess
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip()
print(out)
if int(out.split(",")[1]) < 20000:
    raise SystemExit("GPU 메모리가 20GB 미만입니다. G4/H100/A100 런타임을 선택하세요.")"""),
    ("code", """from google.colab import drive
drive.mount("/content/drive")
import os, shutil
DRIVE_DIR = f"/content/drive/MyDrive/evolve_v4/{RUN_ID}"   # backup every 5 min (safe copy)
RUN_DIR = f"/content/run/{RUN_ID}"                         # live ledger on local disk
os.makedirs(DRIVE_DIR, exist_ok=True); os.makedirs(RUN_DIR, exist_ok=True)
if os.path.exists(f"{DRIVE_DIR}/ledger.sqlite") and not os.path.exists(f"{RUN_DIR}/ledger.sqlite"):
    shutil.copy(f"{DRIVE_DIR}/ledger.sqlite", f"{RUN_DIR}/ledger.sqlite")   # resume
    print("이전 원장을 드라이브에서 복원했습니다")
for stale in (f"{RUN_DIR}/STOP", f"{DRIVE_DIR}/STOP"):
    if os.path.exists(stale): os.remove(stale)
print(RUN_DIR, "->", DRIVE_DIR)"""),
    ("code", f"""%%bash
set -e
rm -rf /content/repo /content/evolve
# only the bundle folder is downloaded (search folders A/B; no C/D holdout)
git clone -q --depth 1 --filter=blob:none --sparse -b {BRANCH} {REPO} /content/repo
git -C /content/repo sparse-checkout set {BUNDLE}
cp -r /content/repo/{BUNDLE} /content/evolve && rm -rf /content/repo
cd /content/evolve && find . -name "__pycache__" -prune -exec rm -rf {{}} +
cat cases/manifest.json
ls population"""),
    ("code", """%%bash
pip install -q vllm threadpoolctl 2>&1 | tail -3
python -c "import vllm; print('vllm', vllm.__version__)\""""),
    ("code", """# 모델 서버: Qwen3.8-27B FP8(+MTP) -> FP8 -> BF16(G4) -> gpt-oss-120b 순으로 시도, 이미 떠 있으면 재사용
import sys
sys.path.insert(0, "/content/evolve")
from agentic_research.evolve_v4 import colab_serve
CHOSEN = colab_serve.serve(MODEL)
MODEL_NAME, CONCURRENCY = CHOSEN[0], CHOSEN[3]"""),
    ("code", """%%bash
# 점검: 번들 안 채점기가 맥과 같은 점수를 내는지 (seed 9158, p10_t6 3300)
cd /content/evolve
for f in population/seed.py population/p10_t6.py; do
  echo -n "$f "; python -m agentic_research.evolve_v4.colab_eval $f | python -c "import json,sys; d=json.load(sys.stdin); print(d['total'], d['totals'])"
done"""),
    ("code", """ENGINE = colab_serve.launch(RUN_DIR, DRIVE_DIR, CHOSEN, HOURS, SEED, EFFORT)
print("- 아래 칸으로 진행 상황을 보세요")"""),
    ("code", """# 진행 상황 (여러 번 실행해도 됨)
import sqlite3, subprocess
db = sqlite3.connect(f"{RUN_DIR}/ledger.sqlite")
print("상태별:", db.execute("SELECT status, COUNT(*) FROM programs GROUP BY status").fetchall())
print("최고 5개:")
for row in db.execute("SELECT id, island, score, totals, substr(idea,1,110) FROM programs WHERE status='ok' "
                      "ORDER BY score LIMIT 5"):
    print(" ", row)
print(subprocess.run(["tail", "-n", "8", f"{RUN_DIR}/engine.log"], capture_output=True, text=True).stdout)
print("engine running:", ENGINE.poll() is None)"""),
    ("code", """# 코드 교체: 돌고 있는 중에 새 코드를 받을 때만 True로 바꾸고 이 칸만 실행.
# 기록은 그대로 이어가고, 더 좋은 모델이 정해져 있으면 모델 서버도 바꿉니다(같으면 1분, 바꾸면 10~30분).
UPDATE_NOW = False
if UPDATE_NOW:
    import subprocess, sys, time, importlib
    subprocess.run(["pkill", "-f", "agentic_research.evolve_v4.colab_engine"])
    while subprocess.run(["pgrep", "-f", "agentic_research.evolve_v4.colab_engine"], capture_output=True).returncode == 0:
        time.sleep(1)
    subprocess.run(\"\"\"set -e
rm -rf /content/repo /content/evolve
git clone -q --depth 1 --filter=blob:none --sparse -b __BRANCH__ __REPO__ /content/repo
git -C /content/repo sparse-checkout set __BUNDLE__
cp -r /content/repo/__BUNDLE__ /content/evolve && rm -rf /content/repo\"\"\", shell=True, check=True, executable="/bin/bash")
    sys.path.insert(0, "/content/evolve")
    from agentic_research.evolve_v4 import colab_serve
    importlib.reload(colab_serve)
    CHOSEN = colab_serve.serve(MODEL)
    MODEL_NAME, CONCURRENCY = CHOSEN[0], CHOSEN[3]
    ENGINE = colab_serve.launch(RUN_DIR, DRIVE_DIR, CHOSEN, HOURS, SEED, EFFORT)
else:
    print("코드 교체 안 함 (UPDATE_NOW = False)")""".replace("__BRANCH__", BRANCH).replace("__REPO__", REPO).replace("__BUNDLE__", BUNDLE)),
    ("code", """# 끝내기: 멈추고 싶을 때만 아래 False를 True로 바꾸고 이 칸만 실행 ('모두 실행' 때는 아무 일도 안 함)
STOP_NOW = False
if STOP_NOW:
    open(f"{RUN_DIR}/STOP", "w").close()  # engine finishes jobs in flight, backs up the ledger, then exits
    print("STOP 파일 생성 — 몇 분 안에 멈춥니다.")
else:
    print("진화 진행 중. 멈추려면 STOP_NOW = True 로 바꾸고 이 칸을 실행하세요.")"""),
]


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    if kind == "md":
        return dict(cell_type="markdown", metadata={}, source=lines)
    return dict(cell_type="code", metadata={}, source=lines, execution_count=None, outputs=[])


if __name__ == "__main__":
    nb = dict(nbformat=4, nbformat_minor=5, cells=[cell(k, t) for k, t in CELLS],
              metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="A100"),
                            kernelspec=dict(name="python3", display_name="Python 3"),
                            language_info=dict(name="python")))
    path = Path(__file__).with_name("evolve_v4_colab.ipynb")
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(path)
