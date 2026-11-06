#!/usr/bin/env bash
# 학습과 감시를 함께 실행한다. tmux 안에서 쓴다 (Claude Code 세션과 분리, 재부팅 시 자동 시작 없음)
#   tmux new-session -d -s pilot "scripts/run_monitored.sh runs/pilot/main --model pilot-42m --data ... "
#   재개: scripts/run_monitored.sh runs/pilot/main --resume runs/pilot/main/last.pt ...
# 첫 인자는 출력 폴더(감시 로그 위치), 나머지는 train.py 인자 그대로
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$1; shift
mkdir -p "$OUT"
echo "[$(date '+%F %T')] 시작: train.py $*" | tee -a "$OUT/run.log"
uv run python train.py "$@" --out-dir "$OUT" >> "$OUT/run.log" 2>&1 &
TRAIN=$!
python3 scripts/monitor.py --out-dir "$OUT" --pid "$TRAIN" >> "$OUT/monitor.log" 2>&1 &
wait "$TRAIN"; CODE=$?
echo "[$(date '+%F %T')] train.py 종료 코드 $CODE (0 정상, 3 NaN, 4 VRAM, 5 데이터 소진, 6 안전 정지)" | tee -a "$OUT/run.log"
wait
exit $CODE
