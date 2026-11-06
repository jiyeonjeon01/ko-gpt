# ko-gpt

무작위 초기화부터 직접 학습하는 한국어 중심 Decoder-only GPT. 다른 회사의 사전학습 가중치를 쓰지 않는다.
설계: 바탕화면 `general-gpt-design/범용대화GPT_설계안_v0.3.md`, 단계별 보고서는 같은 폴더.

## 구조

| 파일 | 역할 |
|---|---|
| `config.py` | `GPTConfig`, 프리셋 `pilot-42m`, `main-110m` |
| `model.py` | RMSNorm, RoPE, SDPA, SwiGLU, 임베딩·LM Head 가중치 공유 |
| `text.py` | `preprocess()` (유일한 정규화 함수), 특수 토큰 16개 (ID 0~15) |
| `tokenizer.py` | 바이트 수준 BPE 학습·로드, `encode()`는 `preprocess()`를 거치는 유일한 입구 |
| `sampler.py` | 출처별 uint16 샤드 혼합 샘플러 (pread + 캐시 버림, 반복 상한, 소진 시 `DataExhausted`) |
| `optim.py` | WSD 학습률, AdamW, BF16·accumulation `train_step()` |
| `train.py` | 사전학습 루프 (출처별 BPB 평가, 원자적 Checkpoint, 재개 검사, `STOP` 파일 안전 정지) |
| `sft.py` | 대화 템플릿과 SFT Loss 마스크 |
| `generate.py` | 고정 프롬프트 이어 쓰기 비교 |
| `bench.py` | 무작위 토큰 처리량·VRAM 측정 |
| `cleaning.py` | 품질 필터, 개인정보 치환, MinHash |
| `scripts/` | 데이터 다운로드·정제·Tokenizer·토큰화, 감시(`monitor.py`), 실행(`run_monitored.sh`, `lr_sweep.sh`) |

데이터(`data/`), 실험 결과(`runs/`), 로그(`logs/`)는 git에 넣지 않는다.

## 명령

```bash
uv sync --group data                 # 설치
uv run pytest -q                     # 테스트 (CUDA 테스트 포함)

# 장시간 학습은 감시와 함께 tmux에서 (Claude Code 세션과 분리)
tmux new-session -d -s <이름> "scripts/run_monitored.sh runs/<실험> --model pilot-42m \
    --data runs/phase0b/data_full.json --micro-batch 8 --grad-accum 32 --max-steps <N> --max-lr 1e-3 ..."
# 안전 정지: runs/<실험>/STOP 파일 생성 → last.pt 저장 후 종료 코드 6
# 재개: STOP 삭제 후 scripts/run_monitored.sh runs/<실험> --resume runs/<실험>/last.pt <같은 학습 인자>

uv run python generate.py runs/<실험>/init.pt runs/<실험>/last.pt --out runs/<실험>/samples.md
python3 scripts/summarize_runs.py runs/<실험> ...
```

## 데이터 준비 (재현)

`scripts/download_sample.py` → `prepare_sample.py` → `tokenizer_compare.py` → `tokenize_sample.py` (Phase 0 표본),
`scripts/plan_0b.py` → `download_0b.py` → `run_0b.sh` (전체: 정제, 코드 근사 중복 제거, 최종 Tokenizer, 토큰화).
