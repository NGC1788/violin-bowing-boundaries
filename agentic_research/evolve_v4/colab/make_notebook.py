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

런타임 → 런타임 유형 변경 → **GPU(A100 또는 H100 권장)** 로 설정한 뒤 위에서부터 차례로 실행하세요."""),
    ("code", """# ===== 설정 (계정마다 바꾸기) =====
RUN_ID = "acct1"      # acct1 / acct2 / acct3
SEED = 1              # 1 / 2 / 3
HOURS = 10            # 진화 시간 (Pro+ 최대 연속 실행 24시간)
MODEL = "auto"        # auto = GPU 메모리에 맞춰 선택
EFFORT = "high"       # gpt-oss 추론 강도: high(질) / medium(속도)"""),
    ("code", """import subprocess, json
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip()
print(out)
gpu_name, vram_mib = out.split(",")[0], int(out.split(",")[1])
if vram_mib >= 70000:
    CANDIDATES = [("openai/gpt-oss-120b", 49152, 8), ("openai/gpt-oss-20b", 49152, 24)]
elif vram_mib >= 20000:
    CANDIDATES = [("openai/gpt-oss-20b", 49152, 24), ("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", 32768, 8)]
else:
    raise SystemExit("GPU 메모리가 20GB 미만입니다. A100/H100/L4 런타임을 선택하세요.")
if MODEL != "auto":
    CANDIDATES = [(MODEL, 32768, 8)]
print("후보 모델:", [c[0] for c in CANDIDATES])"""),
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
    ("code", """import subprocess, time, urllib.request, json
def start(model, max_len):
    log = open("/content/vllm.log", "w")
    proc = subprocess.Popen(["vllm", "serve", model, "--host", "127.0.0.1", "--port", "8000",
                             "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.90",
                             "--max-num-seqs", "32"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):                      # up to 30 min for download + load
        time.sleep(10)
        if proc.poll() is not None:
            return None
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
            return proc
        except Exception:
            pass
    proc.kill()
    return None

SERVER = None
for model, max_len, conc in CANDIDATES:
    print("loading", model, "...")
    SERVER = start(model, max_len)
    if SERVER:
        MODEL_NAME, CONCURRENCY = model, conc
        print("ready:", model)
        break
    print("failed:", model); print(open("/content/vllm.log").read()[-1500:])
assert SERVER, "모든 후보 모델 로드 실패 — 위 로그를 Claude에게 보여주세요."
"""),
    ("code", """%%bash
# 점검: 번들 안 채점기가 맥과 같은 점수를 내는지 (seed 9158, p10_t6 3300)
cd /content/evolve
for f in population/seed.py population/p10_t6.py; do
  echo -n "$f "; python -m agentic_research.evolve_v4.colab_eval $f | python -c "import json,sys; d=json.load(sys.stdin); print(d['total'], d['totals'])"
done"""),
    ("code", """import subprocess
cmd = ["python", "-m", "agentic_research.evolve_v4.colab_engine", "--run-dir", RUN_DIR, "--model", MODEL_NAME,
       "--hours", str(HOURS), "--seed", str(SEED), "--concurrency", str(CONCURRENCY), "--backup-dir", DRIVE_DIR, "--effort", EFFORT]
if "gpt-oss" not in MODEL_NAME:
    cmd += ["--max-tokens", "12000"]
ENGINE = subprocess.Popen(cmd, cwd="/content/evolve", stdout=open(f"{DRIVE_DIR}/stdout.log", "a"),
                          stderr=subprocess.STDOUT)
print("engine started, pid", ENGINE.pid, "- 아래 칸으로 진행 상황을 보세요")"""),
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
    ("code", """# 끝내기: 진행 중인 작업만 마치고 멈춤 (원장은 드라이브에 그대로 남음)
open(f"{RUN_DIR}/STOP", "w").close()  # engine also backs up the ledger to Drive when it stops
print("STOP 파일 생성 — 몇 분 안에 멈춥니다. 다시 시작하려면 STOP 파일을 지우고 엔진 칸을 실행하세요.")"""),
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
