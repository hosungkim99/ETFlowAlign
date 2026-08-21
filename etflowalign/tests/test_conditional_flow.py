"""로컬 검증: reference 조건화 학습/샘플링 경로가 예외 없이 도는가 (Phase 4).

합성 payload(query+reference)로 collate -> 조건부 train step 3회 -> 조건부 ODE 샘플
까지 이어지는지, 출력이 query 크기·유한한지 확인한다.

직접 실행: python -m etflowalign.tests.test_conditional_flow
"""

from __future__ import annotations

import torch

from etflowalign.backbone import EquivariantTransformer
from etflowalign.data.pairs import collate
from etflowalign.flow.prior import get_prior
from etflowalign.sampler import ode_sample
from etflowalign.train import FlowConfig, flow_train_step


def _synthetic_payload(nq: int, nr: int, gen: torch.Generator) -> dict:
    z = torch.randint(1, 10, (nq,), generator=gen)
    pos = torch.randn(nq, 3, generator=gen) * 2.0
    src, dst = [], []
    for i in range(nq - 1):                       # 사슬 결합
        src += [i, i + 1]; dst += [i + 1, i]
    bonds = torch.tensor([src, dst], dtype=torch.long)
    return {
        "z": z, "pos": pos, "bonds": bonds,
        "bond_type": torch.ones(bonds.size(1), dtype=torch.long),
        "ref_z": torch.randint(1, 10, (nr,), generator=gen),
        "ref_pos": torch.randn(nr, 3, generator=gen) * 2.0,
        "mol_id": "syn",
    }


def run():
    gen = torch.Generator().manual_seed(0)
    batch = collate([_synthetic_payload(10, 8, gen), _synthetic_payload(12, 9, gen)])
    assert batch["ref_z"] is not None

    model = EquivariantTransformer(hidden_channels=32, num_layers=2, num_rbf=16,
                                   num_heads=2, cutoff=10.0)
    prior = get_prior("harmonic", scale=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    losses = [flow_train_step(model, batch, prior, opt, grad_clip=10.0) for _ in range(3)]
    assert all(l == l for l in losses), "loss NaN"   # 조건부 학습 경로 정상

    gen_pos = ode_sample(model, batch["z"], batch["bonds"], batch["batch"], prior,
                         n_steps=5, ref_z=batch["ref_z"], ref_pos=batch["ref_pos"],
                         ref_batch=batch["ref_batch"])
    ok = gen_pos.shape == batch["pos"].shape and torch.isfinite(gen_pos).all()
    return {"losses": losses, "gen_shape": tuple(gen_pos.shape), "passed": bool(ok)}


def test_conditional_flow():
    assert run()["passed"]


if __name__ == "__main__":
    res = run()
    print(f"[cond] 조건부 train loss(3스텝) = {[round(l, 3) for l in res['losses']]}")
    print(f"[cond] 조건부 ODE 샘플 출력 = {res['gen_shape']} (query 크기)")
    print(f"[LOCAL COND] {'PASS' if res['passed'] else 'FAIL'} — 조건화 학습/샘플 경로 OK")
