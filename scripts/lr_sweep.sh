#!/usr/bin/env bash
# Phase 1 학습률 비교: pilot-42m, 0.1B 토큰(382 Step) × lr 3개. tmux 안에서 실행
# 같은 seed → 같은 초기 가중치·데이터 순서·평가 창. 다른 것은 --max-lr뿐
# 한 실험이라도 0이 아닌 종료 코드(안전 정지 6 포함)면 남은 실험을 시작하지 않는다
set -uo pipefail
cd "$(dirname "$0")/.."
COMMON="--model pilot-42m --data runs/phase0b/data_full.json --micro-batch 8 --grad-accum 32 --max-steps 382 --warmup-steps 30 --eval-interval 50 --eval-batches 8 --seed 1337"
for LR in 5e-4 1e-3 2e-3; do
  OUT=runs/phase1/lr_$LR
  echo "[$(date '+%F %T')] 시작 lr=$LR → $OUT"
  scripts/run_monitored.sh "$OUT" $COMMON --max-lr "$LR"
  CODE=$?
  echo "[$(date '+%F %T')] 종료 lr=$LR 코드 $CODE"
  if [ "$CODE" -ne 0 ]; then
    echo "중단: lr=$LR 실험이 코드 $CODE로 끝나 남은 실험을 시작하지 않음"
    exit "$CODE"
  fi
done
echo "[$(date '+%F %T')] 학습률 비교 완료"
