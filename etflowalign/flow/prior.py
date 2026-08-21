"""시작 분포(source/prior) 샘플러.

flow matching 은 x0 ~ prior 에서 시작해 목표 x1 로 흐른다.
- GaussianSampler : 단순 등방 가우시안 (엔진 검증용, 게이트 A 기본)
- HarmonicSampler : 결합 그래프 라플라시안 기반. 시작부터 결합된 원자가 가까움
                    (ET-Flow식, 일반화 단계의 production prior)

모든 prior 는 분자별 COM=0 으로 중심화된 좌표를 반환한다(평행이동 불변).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _split_by_batch(batch: Tensor):
    """batch 인덱스 -> 분자별 원자 인덱스 리스트."""
    num_graphs = int(batch.max().item()) + 1
    return [(batch == g).nonzero(as_tuple=True)[0] for g in range(num_graphs)]


class GaussianSampler:
    """x0 ~ N(0, scale^2 I), 분자별 중심화."""

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def sample(self, z: Tensor, bonds: Tensor, batch: Tensor) -> Tensor:
        pos = torch.randn(z.size(0), 3, device=z.device, dtype=torch.float32) * self.scale
        # 분자별 COM 제거
        for idx in _split_by_batch(batch):
            pos[idx] = pos[idx] - pos[idx].mean(dim=0, keepdim=True)
        return pos


class HarmonicSampler:
    """결합 라플라시안 기반 harmonic prior.

    에너지 E(x) = Σ_{(i,j)∈bonds} ||x_i - x_j||^2 에 대응하는 가우시안.
    공분산 Σ = L^+ (라플라시안 유사역). 고유기저에서 std_k = 1/sqrt(λ_k),
    λ≈0(평행이동 모드)은 std=0 으로 두어 자동 중심화된다.

    scale 로 전체 크기를 조절(결합거리 ~1.5Å 근처가 되도록 튜닝).
    """

    def __init__(self, scale: float = 1.0, eps: float = 1e-4):
        self.scale = float(scale)
        self.eps = float(eps)

    def _eig(self, n: int, edges, device):
        """국소 결합 그래프 -> (U, std). std_k = 1/sqrt(λ_k), 영모드는 0."""
        A = torch.zeros(n, n, device=device)
        if edges.numel() > 0:
            A[edges[0], edges[1]] = 1.0
            A = torch.maximum(A, A.T)  # 대칭화
        deg = A.sum(dim=1)
        L = torch.diag(deg) - A
        lam, U = torch.linalg.eigh(L)             # 대칭 -> eigh(오름차순)
        std = torch.zeros_like(lam)
        mask = lam > self.eps
        std[mask] = 1.0 / torch.sqrt(lam[mask])   # 비영 모드만
        return U, std

    def _local_edges(self, idx: Tensor, bonds: Tensor, device):
        """전역 결합에서 이 분자 내부 결합을 국소 인덱스로 변환."""
        gmask = torch.isin(bonds[0], idx) & torch.isin(bonds[1], idx)
        local = {int(a): k for k, a in enumerate(idx.tolist())}
        if gmask.any():
            e = bonds[:, gmask]
            return torch.tensor(
                [[local[int(a)] for a in e[0].tolist()],
                 [local[int(b)] for b in e[1].tolist()]],
                device=device, dtype=torch.long,
            )
        return torch.zeros(2, 0, device=device, dtype=torch.long)

    def sample(self, z: Tensor, bonds: Tensor, batch: Tensor) -> Tensor:
        device = z.device
        out = torch.zeros(z.size(0), 3, device=device)
        for idx in _split_by_batch(batch):
            U, std = self._eig(idx.numel(), self._local_edges(idx, bonds, device), device)
            eps_noise = torch.randn(idx.numel(), 3, device=device)
            out[idx] = (U @ (std[:, None] * eps_noise)) * self.scale
        return out


class PrecomputedHarmonicSampler(HarmonicSampler):
    """고정 배치용: 고유분해(U,std)를 한 번만 계산해 캐시.

    overfit-100 / 대규모 학습에서 매 스텝 eigh 재계산을 피한다.
    build(배치) 후 sample() 은 캐시된 U,std 로 매트멀만 수행(eigh/동기화 없음).
    같은 배치(같은 분자 순서)에 대해서만 유효.
    """

    def __init__(self, scale: float = 1.0, eps: float = 1e-4):
        super().__init__(scale, eps)
        self.cache = None   # list of (idx, U, std)

    def build(self, z: Tensor, bonds: Tensor, batch: Tensor):
        device = z.device
        self.cache = []
        for idx in _split_by_batch(batch):
            U, std = self._eig(idx.numel(), self._local_edges(idx, bonds, device), device)
            self.cache.append((idx, U, std))
        return self

    def sample(self, z: Tensor, bonds: Tensor, batch: Tensor) -> Tensor:
        assert self.cache is not None, "build() 를 먼저 호출"
        out = torch.zeros(z.size(0), 3, device=z.device)
        for idx, U, std in self.cache:
            eps_noise = torch.randn(idx.numel(), 3, device=out.device)
            out[idx] = (U @ (std[:, None] * eps_noise)) * self.scale
        return out


def get_prior(name: str, **kwargs):
    name = name.lower()
    if name == "gaussian":
        return GaussianSampler(**kwargs)
    if name == "harmonic":
        return HarmonicSampler(**kwargs)
    raise ValueError(f"알 수 없는 prior: {name}")
