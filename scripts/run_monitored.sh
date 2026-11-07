#!/usr/bin/env bash
# 학습과 감시를 함께 실행한다. tmux 안에서 쓴다 (Claude Code 세션과 분리, 재부팅 시 자동 시작 없음)
#   tmux new-session -d -s pilot "scripts/run_monitored.sh runs/pilot/main --model pilot-42m --data ... "
#   재개: scripts/run_monitored.sh runs/pilot/main --resume runs/pilot/main/last.pt ...
# 첫 인자는 출력 폴더(감시 로그 위치), 나머지는 train.py 인자 그대로
# SFT는 TRAIN_SCRIPT=sft_train.py (기본 train.py). 둘 외에는 거부한다
# 시작 전에 scripts/preflight.py로 Windows RAM·WSL·GPU·디스크를 확인하고, 실패하면 시작하지 않는다 (종료 코드 8)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$1; shift
SCRIPT=${TRAIN_SCRIPT:-train.py}
case "$SCRIPT" in train.py|sft_train.py) ;; *) echo "TRAIN_SCRIPT는 train.py 또는 sft_train.py: $SCRIPT"; exit 2 ;; esac
mkdir -p "$OUT"
echo "[$(date '+%F %T')] 명령: TRAIN_SCRIPT=$SCRIPT scripts/run_monitored.sh $OUT $*" >> "$OUT/command.txt"
python3 scripts/preflight.py --out-dir "$OUT" -- "$@" 2>&1 | tee -a "$OUT/run.log"
PRE=${PIPESTATUS[0]}
if [ "$PRE" -ne 0 ]; then
  echo "[$(date '+%F %T')] 시작 전 점검 실패 (종료 코드 $PRE). 학습을 시작하지 않음" | tee -a "$OUT/run.log"
  exit "$PRE"
fi
echo "[$(date '+%F %T')] 시작: $SCRIPT $*" | tee -a "$OUT/run.log"
uv run python "$SCRIPT" "$@" --out-dir "$OUT" >> "$OUT/run.log" 2>&1 &
TRAIN=$!
python3 scripts/monitor.py --out-dir "$OUT" --pid "$TRAIN" >> "$OUT/monitor.log" 2>&1 &
wait "$TRAIN"; CODE=$?
echo "[$(date '+%F %T')] $SCRIPT 종료 코드 $CODE (0 정상, 3 NaN, 4 VRAM, 5 데이터 소진, 6 안전 정지, 7 디스크 부족)" | tee -a "$OUT/run.log"
wait
exit $CODE
