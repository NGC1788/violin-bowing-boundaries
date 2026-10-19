#!/bin/zsh
# One-shot: once C1 (2024-04-12_TypeC_sample1.7z) is finished, restart replicate.py so that the reordered
# QUEUE (Part 6 variability_setup next) takes effect. Claude, 2026-10-18, at the user's request.
# Pauses Astra's recover_B1.py (SIGSTOP) during the restart so it cannot mistake the gap for "pipeline finished",
# and resumes it afterwards (SIGCONT). Touches no other process.
set -u
PY=$HOME/violin-bowing-boundaries/.venv-mac/bin/python
T=$HOME/violin-rne/timeres
STATE=$T/replicate_state.json
LOG=$T/restart_part6.log
say() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $LOG; }

say "대기: C1 완료를 기다림"
until $PY -c "import json,sys; s=json.load(open('$STATE')).get('2024-04-12_TypeC_sample1.7z',{}).get('status',''); sys.exit(0 if s=='done' or s.startswith('error') else 1)"; do
  sleep 30
done
say "C1 상태: $($PY -c "import json; print(json.load(open('$STATE'))['2024-04-12_TypeC_sample1.7z']['status'])")"
sleep 20

OLD=$(pgrep -f "python replicate.py$" | tr '\n' ' ')
RB=$(pgrep -f "astra_20261018/recover_B1.py" | head -1)
say "기존 replicate: $OLD / recover_B1: ${RB:-없음}"
[[ -n "$RB" ]] && kill -STOP $RB && say "recover_B1 일시정지"

DL=$(pgrep -f "zenodo_catalog.py download --record" | tr '\n' ' ')
say "다운로드 자식: ${DL:-없음}"
for p in ${=OLD}; do kill $p 2>/dev/null; done
for c in $(pgrep -f "caffeinate -i .*replicate.py"); do kill $c 2>/dev/null; done
for p in ${=DL}; do kill $p 2>/dev/null; done
for i in {1..30}; do
  if ! pgrep -f "python replicate.py$" >/dev/null && ! pgrep -f "zenodo_catalog.py download --record 17782542" >/dev/null; then break; fi
  sleep 1
done
say "종료 확인: replicate $(pgrep -f 'python replicate.py$' | wc -l | tr -d ' ')개 남음"
find $HOME/violin-rne/data -name "*.download.lock" -print -delete >> $LOG 2>&1
$PY - <<EOF >> $LOG 2>&1
import json
p = "$STATE"; s = json.load(open(p))
for k, v in s.items():
    if str(v.get("status", "")).startswith("error"):
        print("pending으로 되돌림:", k, v["status"]); v["status"] = "pending"
json.dump(s, open(p, "w"), indent=1)
EOF

cd $T
nohup caffeinate -i $PY replicate.py >> replicate.log 2>&1 &
sleep 15
say "새 replicate: $(pgrep -f 'python replicate.py$' | tr '\n' ' ')"
for i in {1..40}; do
  if grep -q "다운로드 시작 2024-10-23_variability_setup.7z" replicate.log; then break; fi
  sleep 3
done
say "로그 끝: $(tail -1 replicate.log)"
[[ -n "${RB:-}" ]] && kill -CONT $RB && say "recover_B1 재개"
say "완료"
