"""flow-matching 학습 루프 (Phase 2: 조건 없음).

한 스텝:
  t ~ U(0,1) (분자별) -> x0 ~ prior -> x_t=(1-t)x0+t·x1 -> v_θ(x_t,t)
  loss = batchwise_l2(v_θ, x1-x0)

안정화: clip_grad_norm + nan_to_num (과거 NaN 크래시 방지).
best 손실 즉시 보관, NaN step 은 이전 best 로 롤백.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from etflowalign.flow.loss import batchwise_l2_loss
from etflowalign.flow.path import (
    center_by_reference,
    center_pos,
    interpolate,
    kabsch_align_source_to_target,
    sample_time,
    target_velocity,
)
from etflowalign.flow.prior import get_prior


@dataclass
class FlowConfig:
    prior: str = "gaussian"      # "gaussian" | "harmonic"
    prior_scale: float = 1.0
    lr: float = 2e-4
    grad_clip: float = 10.0
    steps: int = 2000
    log_every: int = 200
    use_ema: bool = True         # 가중치 지수이동평균(생성 품질↑, FM 표준)
    ema_decay: float = 0.999
    kabsch_source_align: bool = True  # 비조건 학습 시 source->target 회전정렬(등변 필수)
    force_conditioned_align: bool = False  # 진단용: 조건부에서도 정렬 강제(회전 제거)
    n_micro: int = 1             # 스텝당 (x0,t) 샘플 수(K>1=분산↓, B1)


class EMA:
    """부동소수 파라미터의 지수이동평균. 생성 시 EMA 가중치가 더 매끄럽다."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k])
        model.load_state_dict(sd)


def flow_train_step(model, batch, prior, optimizer, grad_clip: float,
                    kabsch_align: bool = False, force_conditioned_align: bool = False,
                    n_micro: int = 1):
    """단일 배치 학습 스텝. 반환: loss 값(float).

    batch 에 ref_z 가 있으면 reference 조건화 모드로 동작한다.
    kabsch_align: 비조건 모드에서 source(x0)를 target(x1)에 회전정렬(등변 필수).
    n_micro: 스텝당 (x0,t) 샘플 수. K>1 이면 grad 누적으로 분산↓ (B1).
    """
    model.train()
    z, x1, bonds, b = batch["z"], batch["pos"], batch["bonds"], batch["batch"]
    B = int(b.max().item()) + 1
    conditioned = batch.get("ref_z") is not None

    if conditioned:
        ref_z, ref_pos, ref_batch = batch["ref_z"], batch["ref_pos"], batch["ref_batch"]
        x1, ref_pos = center_by_reference(x1, b, ref_pos, ref_batch, B)  # ref COM 기준
    else:
        x1 = center_pos(x1, b)                      # 타깃 중심화

    prior_b = batch.get("_prior", prior)            # 배치별 캐시 prior 우선
    optimizer.zero_grad()
    total = 0.0
    for _ in range(n_micro):                        # 여러 (x0,t) 평균 -> 분산↓
        t = sample_time(B, device=z.device)
        x0 = prior_b.sample(z, bonds, b)            # query 시작점
        if kabsch_align and (not conditioned or force_conditioned_align):
            # 무작위 방향 x0 를 x1 방향에 정렬 -> 등변 모델이 모양만 학습
            x0 = kabsch_align_source_to_target(x0, x1, b, B)
        xt = interpolate(x0, x1, t, b)
        u = target_velocity(x0, x1)                 # 목표 속도
        if conditioned:
            v = model(z, xt, t, b, ref_z, ref_pos, ref_batch)
        else:
            v = model(z, xt, t, b)
        loss = batchwise_l2_loss(v, u, b)
        (loss / n_micro).backward()                 # grad 누적
        total += float(loss.detach())

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # clip 은 NaN 을 못 막음 -> 잔여 NaN/Inf grad 를 0 으로 (과거 교훈)
    for p in model.parameters():
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    optimizer.step()
    return total / n_micro


def train_flow(model: nn.Module, batches, cfg: FlowConfig, verbose: bool = True, prior=None):
    """batches: 매 스텝 사용할 배치 dict 의 리스트(작으면 순환). 반환: best state_dict.

    prior 를 지정하면 그걸 사용(예: PrecomputedHarmonicSampler), 아니면 cfg 로 생성.
    """
    if prior is None:
        prior = get_prior(cfg.prior, scale=cfg.prior_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None
    nan_streak = 0

    for step in range(cfg.steps):
        batch = batches[step % len(batches)]
        loss = flow_train_step(model, batch, prior, optimizer, cfg.grad_clip,
                               kabsch_align=cfg.kabsch_source_align,
                               force_conditioned_align=cfg.force_conditioned_align,
                               n_micro=cfg.n_micro)

        if loss != loss or loss == float("inf"):    # NaN/Inf 방어
            nan_streak += 1
            model.load_state_dict(best_state)        # 이전 best 로 롤백
            if nan_streak > 20:
                raise RuntimeError("NaN 20회 초과 — 학습 중단")
            continue
        nan_streak = 0

        if ema is not None:
            ema.update(model)
        if loss < best_loss:
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"[step {step:6d}] loss={loss:.4f}  best={best_loss:.4f}")

    # 생성용 가중치: EMA 우선(더 매끄러움), 없으면 best
    if ema is not None:
        ema.copy_to(model)
    else:
        model.load_state_dict(best_state)
    return model.state_dict(), best_loss
