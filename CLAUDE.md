# ko-gpt 작업 규칙

- 다른 회사의 사전학습 가중치·LoRA 미세조정 금지. 무작위 초기화부터 직접 학습한다
- `~/dev/mini-gpt-from-scratch` (이전 TinyStories 프로젝트)와 그 Checkpoint는 읽기만 한다
- 데이터는 공개 가능 A 그룹만 기본 사용 (B 그룹 보류). 다운로드 전에 파일 목록·용량을 보고하고 승인받는다
- 단계(Phase)마다 사용자 승인 후 진행한다. 다음 Phase를 임의로 시작하지 않는다
- 장시간 작업은 `tmux` + `scripts/run_monitored.sh`로 실행한다. Claude Code 백그라운드 셸은 PC 메모리 부족 시 강제 종료된다
- Docker Desktop·다른 프로젝트·다른 WSL 배포판은 이 PC에서 함께 쓰는 자원이다. GPT 학습 때문에 종료·중단·설정 변경하지 않는다. `wsl --shutdown`도 실행하지 않는다. 자원이 부족하면 GPT 학습 시작만 거부하고 이유를 보고한다 (`scripts/preflight.py`)
- 이상 시 학습 설정을 자동으로 바꾸거나 데이터를 건너뛰지 않는다. 기록하고 멈춘다. 재부팅 후 자동 재개하지 않는다
- 기존 실험 결과(`runs/`)를 덮어쓰지 않는다
- 커밋은 `~/.claude/skills/to-new-session/references/commit-convention.md` 규칙 (Windows: `C:\Users\jiyeo\.claude\...`)
- 추정은 (추정), 측정은 (측정)으로 구분해서 보고한다
- 실험 결과에는 실행 횟수(단일 실행인지)를 적는다. 같은 설정·seed라도 GPU 비결정성으로 결과가 달라진다 (Phase 1 측정: 같은 설정 두 실행의 step 250 한국어 BPB 차 0.059). 이보다 작은 차이는 반복 실행 없이 결론으로 쓰지 않는다
- 평가 세트(`eval/chat120`, `eval/prompts_ko20.json`)는 SFT 학습 데이터에 넣지 않는다. 고정본은 모델 응답을 본 뒤 고치지 않는다
