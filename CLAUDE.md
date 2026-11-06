# ko-gpt 작업 규칙

- 다른 회사의 사전학습 가중치·LoRA 미세조정 금지. 무작위 초기화부터 직접 학습한다
- `~/dev/mini-gpt-from-scratch` (이전 TinyStories 프로젝트)와 그 Checkpoint는 읽기만 한다
- 데이터는 공개 가능 A 그룹만 기본 사용 (B 그룹 보류). 다운로드 전에 파일 목록·용량을 보고하고 승인받는다
- 단계(Phase)마다 사용자 승인 후 진행한다. 다음 Phase를 임의로 시작하지 않는다
- 장시간 작업은 `tmux` + `scripts/run_monitored.sh`로 실행한다. Claude Code 백그라운드 셸은 PC 메모리 부족 시 강제 종료된다
- 이상 시 학습 설정을 자동으로 바꾸거나 데이터를 건너뛰지 않는다. 기록하고 멈춘다. 재부팅 후 자동 재개하지 않는다
- 기존 실험 결과(`runs/`)를 덮어쓰지 않는다
- 커밋은 `~/.claude/skills/to-new-session/references/commit-convention.md` 규칙 (Windows: `C:\Users\jiyeo\.claude\...`)
- 추정은 (추정), 측정은 (측정)으로 구분해서 보고한다
