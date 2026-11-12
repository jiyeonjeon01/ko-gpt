"""Checkpoint에서 모델 가중치만 적은 메모리로 읽는다 (SFT 시작·평가·생성용)

사전학습 Checkpoint(110M 약 1.32GB)에는 모델 가중치(0.44GB) 말고도 Optimizer 상태(0.88GB)·Sampler·RNG가 들어 있다.
torch.load로 통째로 읽으면 전부 메모리에 올라간다. 여기서는 다음과 같이 읽는다.

- torch.load(mmap=True, weights_only=True): 텐서를 파일에 대응시키기만 하고 바로 읽지 않는다.
- 필요한 항목(model, model_config, run, step, tokens_seen, kind, sft)만 새 dict로 옮기고, 나머지 참조는 지운다.
  - mmap이면 Optimizer가 저절로 빠진다고 가정하지 않는다. 필요한 키만 골라 담는다.
- 가중치를 모델에 복사한 뒤에는 호출한 쪽이 반환값을 지우고 release()로 그 파일의 캐시를 돌려준다 (filecache.py).
- 학습 재개(--resume)에는 쓰지 않는다. 재개에는 Optimizer 상태가 필요하다.
"""

from pathlib import Path

import torch

from filecache import drop_cache
from tokenizer import file_sha256

KEEP = ("model", "model_config", "run", "step", "tokens_seen", "kind", "sft")


def load_weights(path: Path) -> dict:
    """KEEP 키만 담은 dict. model 값은 파일에 대응된 텐서다 (load_state_dict가 복사하면 지워도 된다)"""
    ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model" not in ck or "model_config" not in ck:
        raise ValueError(f"{path}: model·model_config가 없는 Checkpoint")
    out = {k: ck[k] for k in KEEP if k in ck}
    ck.clear()  # Optimizer·Sampler·RNG 참조를 바로 끊는다
    del ck
    return out


def release(path: Path) -> dict:
    """가중치를 다 옮기고 반환값을 지운 뒤 부른다: 그 Checkpoint 파일 하나의 캐시만 돌려준다.
    캐시 반환은 최적화일 뿐이라 실패해도 예외를 내지 않고 이유만 돌려준다 (로딩·학습을 막지 않음)"""
    try:
        return drop_cache(path)
    except Exception as e:  # noqa: BLE001
        return {str(path): f"{type(e).__name__}: {e}"}


def checkpoint_sha256(path: Path) -> str:
    """해시 계산 후 그 파일의 캐시만 돌려준다 (1.3GB 파일 전체가 WSL 캐시에 남지 않게)"""
    digest = file_sha256(path)
    release(path)
    return digest


def meta_of(w: dict) -> dict:
    """가중치를 뺀 작은 정보"""
    return {k: v for k, v in w.items() if k != "model"}
