"""감시 판단: 일시적인 값 한 번으로는 멈추지 않고, 지속되면 멈춘다"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from monitor import Judge


def sample(win=6000, swap=200, gpu_used=8000, temp=60):
    return {"win_avail_mb": win, "swap_used_mb": swap, "gpu_used_mb": gpu_used, "gpu_total_mb": 12288,
            "gpu_temp_c": temp}


def run(samples):
    j = Judge()
    return [j.update(s)[0] for s in samples]


def test_normal_never_stops():
    assert run([sample()] * 50) == [None] * 50


def test_single_low_ram_spike_does_not_stop():
    seq = [sample()] * 5 + [sample(win=500)] * 3 + [sample()] * 10
    assert all(r is None for r in run(seq))


def test_sustained_low_ram_stops_on_fourth_sample():
    out = run([sample()] * 3 + [sample(win=800)] * 4)
    assert out[:6] == [None] * 6 and "Windows" in out[6]


def test_unknown_windows_ram_is_not_treated_as_low():
    assert all(r is None for r in run([sample(win=None)] * 10))


def test_swap_growth_over_five_minutes_stops():
    seq = [sample(swap=200 + 150 * i) for i in range(10)]  # 10회 동안 1,350MB 증가
    out = run(seq)
    assert out[-1] is not None and "swap" in out[-1] and all(r is None for r in out[:-1])


def test_swap_absolute_needs_two_samples():
    out = run([sample(swap=4200), sample(swap=4300)])
    assert out[0] is None and "swap" in out[1]


def test_gpu_near_full_and_hot_need_persistence():
    out = run([sample(gpu_used=12100)] * 3 + [sample()] + [sample(gpu_used=12100)] * 4)
    assert out[:7] == [None] * 7 and "GPU 메모리" in out[7]
    out = run([sample(temp=91)] * 4)
    assert out[:3] == [None] * 3 and "온도" in out[3]


def test_warnings_do_not_stop():
    j = Judge()
    reason, warnings = j.update(sample(win=1500, temp=86))
    assert reason is None and len(warnings) == 2
