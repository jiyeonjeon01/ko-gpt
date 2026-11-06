#!/usr/bin/env bash
# Phase 0b 처리: 정제 → 코드 근사 중복 제거 → 최종 Tokenizer → 토큰화 (다운로드 완료 후 실행)
# 단계가 실패하면 그 자리에서 멈춘다. 단계별 로그: logs/0b_<단계>.log
set -euo pipefail
cd "$(dirname "$0")/.."
step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] 시작: $name"
  /usr/bin/time -v "$@" > "logs/0b_${name}.log" 2>&1 || { echo "[$(date '+%F %T')] 실패: $name (logs/0b_${name}.log)"; exit 1; }
  echo "[$(date '+%F %T')] 완료: $name ($(grep -E 'Elapsed|Maximum resident' "logs/0b_${name}.log" | tr -s ' ' | tr '\n' ' '))"
}
step prepare uv run --group data python scripts/prepare_sample.py --stage full
step dedup_code uv run --group data python scripts/dedup_near.py --clean data/full/clean --source P8a_opc_algorithmic
step tokenizer uv run --group data python scripts/train_final_tokenizer.py
step tokenize uv run --group data python scripts/tokenize_full.py
echo "[$(date '+%F %T')] Phase 0b 처리 완료"
