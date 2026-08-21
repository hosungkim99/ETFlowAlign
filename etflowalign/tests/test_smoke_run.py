"""로컬 최소 검증: 코드 경로가 예외 없이 도는가 (게이트 판정 아님).

노트북에서 소음/부하 없이 몇 초 안에 끝나도록 초소형 모델·3스텝만 돌린다.
실제 학습/게이트 판정은 A10 서버에서 (scripts/03_smoke_generate.py).

직접 실행: python -m etflowalign.tests.test_smoke_run
"""

from __future__ import annotations

import importlib.util
import os

import torch

_SMOKE = os.path.join(os.path.dirname(__file__), "..", "scripts", "03_smoke_generate.py")


def _load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_mod", os.path.abspath(_SMOKE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_run(prior: str):
    m = _load_smoke()
    # 초소형 설정: 순식간에 끝남
    return m.run(prior_name=prior, num_mols=2, steps=3, n_sample_steps=3,
                 device="cpu", hidden_channels=32, num_layers=2,
                 num_rbf=16, num_heads=2)


def test_smoke_gaussian():
    torch.manual_seed(0)
    assert _tiny_run("gaussian") in (True, False)  # 예외 없이 완주하면 통과


def test_smoke_harmonic():
    torch.manual_seed(0)
    assert _tiny_run("harmonic") in (True, False)


if __name__ == "__main__":
    for p in ("gaussian", "harmonic"):
        _tiny_run(p)
        print(f"[smoke] prior={p}: 코드 경로 정상 실행 OK")
    print("[LOCAL SMOKE] PASS — 실제 학습/게이트는 서버에서")
