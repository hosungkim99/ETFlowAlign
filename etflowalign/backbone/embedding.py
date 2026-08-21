"""입력 임베딩: 원자종 + flow-matching 시간 t + reference 플래그.

Phase 4: query/reference 노드를 is_ref 플래그(불변 스칼라)로 구분한다.
reference 는 위치(상대 기하)로만 조건에 들어가고, 원자종/플래그는 불변이라
등변성을 깨지 않는다(SPEC §5.4).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalTime(nn.Module):
    """연속 시간 t in [0,1] 를 sinusoidal 임베딩으로."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:  # t: [B]
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(-scale * torch.arange(half, device=t.device, dtype=t.dtype))
        args = t[:, None] * freqs[None, :] * 1000.0  # t 를 넓은 주파수 범위로 확장
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
        if emb.size(-1) < self.dim:  # dim 이 홀수면 0 패딩
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), self.dim - emb.size(-1))], dim=-1)
        return emb


class AtomTimeEmbedding(nn.Module):
    """원자종 z -> 스칼라 피처 h, 시간 t 를 분자별로 더하고, is_ref 플래그를 얹는다."""

    def __init__(self, hidden_channels: int, max_z: int = 100):
        super().__init__()
        self.atom_emb = nn.Embedding(max_z, hidden_channels)
        self.ref_flag_emb = nn.Embedding(2, hidden_channels)   # 0=query, 1=reference
        nn.init.zeros_(self.ref_flag_emb.weight)               # 초기엔 무영향(Phase2 호환)
        self.time_mlp = nn.Sequential(
            SinusoidalTime(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, z: Tensor, t: Tensor, batch: Tensor, is_ref: Tensor = None) -> Tensor:
        """z: [N] long, t: [B] float, batch: [N] long, is_ref: [N] long|None -> h: [N, F]."""
        h = self.atom_emb(z)                 # [N, F]
        t_emb = self.time_mlp(t)             # [B, F]
        h = h + t_emb[batch]                 # 분자별 시간 임베딩을 원자에 브로드캐스트
        if is_ref is not None:
            h = h + self.ref_flag_emb(is_ref)  # query/reference 구분 (불변)
        return h
