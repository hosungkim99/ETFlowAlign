"""거리 특징화: expnorm RBF + cosine cutoff 포락선.

TorchMD-Net 의 ExpNormalSmearing / CosineCutoff 을 self-contained 로 재구현.
거리 |r_ij| (불변 스칼라)를 여러 채널로 펼쳐 에지 특징을 만든다.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CosineCutoff(nn.Module):
    """cutoff 반경에서 부드럽게 0 으로 감쇠하는 포락선.

    0.5 * (cos(pi * r / cutoff) + 1),  r >= cutoff 에서는 0.
    어텐션 가중치에 곱해져 먼 이웃의 기여를 매끄럽게 없앤다.
    """

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, dist: Tensor) -> Tensor:
        out = 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0)
        return out * (dist < self.cutoff).to(dist.dtype)


class ExpNormalSmearing(nn.Module):
    """expnorm 방사 기저 함수 (TorchMD-Net 기본).

    exp(-beta * (exp(alpha*(cutoff_lower - r)) - mean)^2) * cutoff(r)
    means 는 [exp(-cutoff_upper+cutoff_lower), 1] 구간에 균등 배치.
    """

    def __init__(
        self,
        cutoff: float = 10.0,
        num_rbf: int = 64,
        cutoff_lower: float = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.cutoff_lower = float(cutoff_lower)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / (cutoff - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start = math.exp(-self.cutoff + self.cutoff_lower)
        means = torch.linspace(start, 1.0, self.num_rbf)
        beta_val = (2.0 / self.num_rbf * (1.0 - start)) ** -2
        betas = torch.full((self.num_rbf,), beta_val)
        return means, betas

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)  # [E, 1]
        env = self.cutoff_fn(dist)  # [E, 1]
        smeared = torch.exp(
            -self.betas * (torch.exp(self.alpha * (self.cutoff_lower - dist)) - self.means) ** 2
        )
        return env * smeared  # [E, num_rbf]
