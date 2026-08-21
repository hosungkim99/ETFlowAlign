"""ODE 샘플러: x0 ~ prior 에서 시작해 v_θ 를 적분하여 x1 생성.

dx/dt = v_θ(x, t),  t: 0 -> 1,  Euler 적분 (few-step).
포켓 UFF 가이던스 훅은 옵션 (Phase 7 에서 연결).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def ode_sample(
    model: nn.Module,
    z: Tensor,
    bonds: Tensor,
    batch: Tensor,
    prior,
    n_steps: int = 50,
    guidance_fn=None,
    x0: Tensor = None,
    ref_z: Tensor = None,
    ref_pos: Tensor = None,
    ref_batch: Tensor = None,
) -> Tensor:
    """ODE 샘플링. ref_* 가 주어지면 reference 조건화 모드(ref 고정).

    model: forward(z, pos, t, batch[, ref_z, ref_pos, ref_batch]) -> v [Nq,3]
    x0:    지정 시 그 시작점 사용, 아니면 prior 에서 샘플
    반환:  생성된 query 좌표 x1 [Nq, 3]  (reference 는 고정, 반환 안 함)
    """
    model.eval()
    num_graphs = int(batch.max().item()) + 1
    conditioned = ref_z is not None
    x = prior.sample(z, bonds, batch) if x0 is None else x0.clone()
    dt = 1.0 / n_steps

    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((num_graphs,), t_val, device=z.device, dtype=x.dtype)
        if conditioned:
            v = model(z, x, t, batch, ref_z, ref_pos, ref_batch)
        else:
            v = model(z, x, t, batch)
        if guidance_fn is not None:
            v = v + guidance_fn(x, t, batch)  # 포켓 UFF 등 (Phase 7)
        x = x + dt * v

    return x
