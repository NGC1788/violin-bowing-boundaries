"""Model server and engine launch for the Colab notebook, kept in one module so a pasted cell can call it.

Model choice (checked 2026-11-05 on Hugging Face and recipes.vllm.ai): Qwen3.8-27B (Apache-2.0, Aug 2026)
is the newest open coding model that fits one Colab GPU. The Flash/MoE releases of the same period
(Qwen3.8-Flash-Next 360 GB, GLM-5.3-Flash 328 GB, DeepSeek-V4.1-Flash 510 GB) do not fit. gpt-oss-120b
stays as the fallback. Sampling follows the model card (thinking mode), set in colab_engine.post_chat.
"""
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("EVOLVE_PORT", "8000"))
URL = f"http://127.0.0.1:{PORT}/v1"
LOG = os.environ.get("EVOLVE_VLLM_LOG", "/content/vllm.log")
CODE = os.environ.get("EVOLVE_CODE_DIR", "/content/evolve")
ENGINE = "agentic_research.evolve_v4.colab_engine"


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    name, total, used = [x.strip() for x in out.split(",")]
    return name, int(total), int(used)


def qwen_args(mtp):
    args = ["--reasoning-parser", "qwen3", "--language-model-only"]
    if mtp:  # built-in multi-token-prediction draft head (vLLM recipe)
        args += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp})]
    return args


def candidates(model="auto"):
    """Tried in order: (model, max_len, max_num_seqs, engine concurrency, extra vllm args, engine --max-tokens)."""
    _, vram, _ = gpu()
    out = []
    if vram >= 70000:  # G4 (RTX PRO 6000, 96 GB), H100 / A100 80 GB
        out += [("Qwen/Qwen3.8-27B-FP8", 65536, 12, 8, qwen_args(3), None),
                ("Qwen/Qwen3.8-27B-FP8", 65536, 12, 8, qwen_args(0), None)]
        if vram >= 90000:  # the recipe's own G4 setting: BF16 weights, 5 draft tokens
            out += [("Qwen/Qwen3.8-27B", 65536, 8, 8, qwen_args(5), None)]
        out += [("openai/gpt-oss-120b", 49152, 32, 8, [], None)]
    elif vram >= 20000:
        out += [("openai/gpt-oss-20b", 49152, 32, 24, [], None),
                ("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", 32768, 32, 8, [], 12000)]
    else:
        raise SystemExit("GPU 메모리가 20GB 미만입니다. G4/H100/A100 런타임을 선택하세요.")
    if model != "auto":
        out = [c for c in out if c[0] == model] or [(model, 32768, 32, 8, [], 12000)]
    return out


def version(pkg):
    try:
        return tuple(int(x) for x in re.findall(r"\d+", md.version(pkg))[:2])
    except md.PackageNotFoundError:
        return (0, 0)


def ensure_packages():
    """Qwen3.8 needs vLLM >= 0.17 and transformers >= 5.8 (vLLM recipe). Upgrade only when older."""
    if os.environ.get("EVOLVE_SKIP_PIP"):  # local tests
        return
    need = (["vllm"] if version("vllm") < (0, 17) else []) + \
           (["transformers>=5.8.0"] if version("transformers") < (5, 8) else [])
    if need:
        print("upgrading", need, "(몇 분 걸림)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *need], check=False)
    print("vllm", md.version("vllm"), "transformers", md.version("transformers"))


def served():
    try:
        return json.loads(urllib.request.urlopen(URL + "/models", timeout=5).read())["data"][0]["id"]
    except Exception:
        return None


def alive(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def stop_server():
    subprocess.run(["pkill", "-f", "vllm serve"])
    subprocess.run(["pkill", "-f", "VLLM::"])  # engine-core children
    for _ in range(90):  # wait until the GPU memory is really free
        if not alive("vllm serve") and gpu()[2] < 3000:
            return
        time.sleep(2)


def start(c):
    model, max_len, seqs, _, extra, _ = c
    stop_server()
    log = open(LOG, "w")
    proc = subprocess.Popen([os.environ.get("EVOLVE_VLLM_BIN", "vllm"), "serve", model, "--host", "127.0.0.1",
                             "--port", str(PORT), "--max-model-len", str(max_len),
                             "--gpu-memory-utilization", "0.90", "--max-num-seqs", str(seqs), *extra],
                            stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):  # up to 40 min: download + load + compile
        time.sleep(float(os.environ.get("EVOLVE_POLL", "10")))
        if proc.poll() is not None:
            return False
        if served():
            return True
    proc.kill()
    return False


def serve(model="auto"):
    """Start (or reuse) the best model that loads; returns its candidate tuple."""
    cands = candidates(model)
    for _ in range(120):  # an earlier server may still be loading: wait for it
        if served() or not alive("vllm serve"):
            break
        time.sleep(10)
    if served() == cands[0][0]:
        print("already running:", cands[0][0])
        return cands[0]
    if cands[0][0].startswith("Qwen/Qwen3.8"):
        ensure_packages()
    for c in cands:
        print("loading", c[0], " ".join(c[4]), "... (처음엔 다운로드 포함 10~30분)", flush=True)
        if start(c):
            print("ready:", c[0])
            return c
        print("failed:", c[0])
        print(Path(LOG).read_text()[-1500:])
    raise SystemExit("모든 후보 모델 로드 실패 — 위 로그를 Claude에게 보여주세요.")


def stop_engine():
    subprocess.run(["pkill", "-f", ENGINE])
    while alive(ENGINE):
        time.sleep(1)


def launch(run_dir, drive_dir, c, hours, seed, effort="medium"):
    """Restart the evolution engine on the served model; the ledger in run_dir carries over."""
    stop_engine()
    cmd = ["python", "-m", ENGINE, "--run-dir", run_dir, "--model", c[0], "--hours", str(hours),
           "--seed", str(seed), "--concurrency", str(c[3]), "--backup-dir", drive_dir, "--effort", effort,
           "--base-url", URL]
    if c[5]:
        cmd += ["--max-tokens", str(c[5])]
    engine = subprocess.Popen(cmd, cwd=CODE, stdout=open(f"{drive_dir}/stdout.log", "a"), stderr=subprocess.STDOUT)
    time.sleep(float(os.environ.get("EVOLVE_WAIT", "10")))
    log = Path(run_dir) / "engine.log"
    print(log.read_text().splitlines()[-1] if log.exists() else "(engine.log not yet written)")
    print("engine pid", engine.pid, "model", c[0])
    return engine
