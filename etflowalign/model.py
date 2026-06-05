"""ETFlowAlign 핵심 모델 정의.

이 모듈은 정렬을 위한 최소한의 구체적인 E(3)-등변 벡터 필드 모델을 구현한다.
아키텍처는 외부 래퍼 없이 이 저장소만으로 전체 설계를 이해할 수 있도록
의도적으로 간결하게 구성되어 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .utils import safe_norm, segment_mean


@dataclass
class AlignmentBatch:
    """ETFlowAlign를 위한 최소한의 독립형 배치 컨테이너.

    한 줄 요약:
        쿼리 리간드/레퍼런스/포켓의 좌표·원자타입·그래프인덱스 등 모델 입력에
        필요한 모든 텐서를 하나로 묶는 데이터 클래스.

    생성이유:
        모델(ETFlowAlignModel)은 단일 미니배치에 여러 복합체(그래프)를 packed
        형태로 받아 처리한다. 좌표/원자타입/배치인덱스/조건정보를 개별 인자로
        넘기면 시그니처가 비대해지고 옵션(레퍼런스·포켓·결합정보)의 유무를
        다루기 어렵다. 이를 하나의 명시적 컨테이너로 묶어 전달/조건 분기를
        단순화하기 위해 만들어졌다.

    역할:
        data.load_alignment_batch_from_pt / dataset.* 가 구성하여 모델 forward와
        flow_matching, sampler 전반에서 단일 입력 객체로 사용된다. 선택 필드가
        None이면 해당 조건(레퍼런스/포켓/결합)이 비활성임을 의미한다.

    파이프라인 단계:
        2단계(데이터 로딩)에서 생성되어 3단계(모델 본체)·4단계(flow matching)·
        7단계(샘플링)의 입력으로 사용된다.

    Attributes:
        query_pos: 쿼리 리간드 좌표, 형상 ``[Nq, 3]``.
        query_atom_type: 쿼리 원자의 정수형 원자 타입 ID ``[Nq]``.
        query_batch: 각 쿼리 원자의 그래프 인덱스 ``[Nq]``.
        reference_pos: 레퍼런스 리간드 좌표 ``[Nr, 3]``.
        reference_atom_type: 선택적 레퍼런스 원자 타입 ID ``[Nr]``.
        reference_batch: 각 레퍼런스 원자의 그래프 인덱스 ``[Nr]``.
        pocket_pos: 선택적 포켓/수용체 포인트 좌표.
    """

    query_pos: Tensor
    query_atom_type: Tensor
    query_batch: Tensor
    reference_pos: Optional[Tensor] = None
    reference_atom_type: Optional[Tensor] = None
    reference_batch: Optional[Tensor] = None
    pocket_pos: Optional[Tensor] = None
    pocket_batch: Optional[Tensor] = None
    pocket_atom_type: Optional[Tensor] = None
    query_node_attr: Optional[Tensor] = None
    query_bond_index: Optional[Tensor] = None
    query_bond_length: Optional[Tensor] = None
    reference_node_attr: Optional[Tensor] = None


class SimpleTimeEmbedding(nn.Module):
    """[0, 1] 구간의 연속 플로우 시간 t에 대한 사인 임베딩.

    한 줄 요약:
        스칼라 플로우 시간 t를 고차원 사인/코사인 임베딩으로 변환한 뒤 MLP로
        투영하는 모듈.

    생성이유:
        flow matching에서 속도 필드는 시간 t에 의존한다(v_theta(x_t, t)). 스칼라
        t를 그대로 쓰면 네트워크가 시간 의존성을 표현하기 어려우므로, 여러
        주파수의 사인/코사인으로 펼쳐 풍부한 시간 표현을 제공하기 위해 만들어졌다.

    역할:
        그래프별 시간 t를 받아 노드 피처와 결합될 시간 임베딩을 생성한다.

    주요 변수(속성):
        dim: 임베딩 차원 수.
        proj: 사인 임베딩을 동일 차원으로 변환하는 2층 MLP(Linear-SiLU-Linear).

    파이프라인 단계:
        3단계(모델 본체). forward 초반에서 t_graph를 임베딩하여 in_proj 입력에
        포함시킨다.
    """

    def __init__(self, dim: int) -> None:
        """시간 임베딩 모듈을 초기화한다.

        생성하는 주요 속성:
            dim: 임베딩(출력) 차원.
            proj: 사인/코사인 임베딩을 비선형 변환하는 MLP.

        Args:
            dim (int): 시간 임베딩의 출력 차원.
        """
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        """각 그래프에 대한 플로우 시간을 임베딩한다.

        한 줄 요약:
            스칼라 시간 t를 다주파수 사인/코사인 벡터로 펼친 뒤 MLP로 투영한다.

        생성이유:
            시간 의존 속도 필드를 학습하려면 t를 네트워크가 다루기 쉬운 고차원
            표현으로 변환해야 한다.

        역할:
            그래프별 시간 텐서를 받아 ``[B, dim]`` 시간 임베딩을 반환한다.

        메커니즘:
            dim/2개의 주파수를 기하급수적으로 생성하고 각 주파수에 대해 sin/cos를
            계산해 이어 붙인다(half==0이면 t를 그대로 반환). 차원이 모자라면
            0으로 패딩한 뒤 proj MLP를 통과시킨다.

        Args:
            t (Tensor): ``[0, 1]`` 구간의 연속 시간, 형상 ``[B]``.

        Returns:
            Tensor: 형상 ``[B, dim]``의 시간 임베딩.

        파이프라인 단계:
            3단계(모델 본체). forward에서 t_graph 임베딩에 사용된다.
        """
        half = self.dim // 2
        if half == 0:
            return t[:, None]
        freq_exp = -torch.log(torch.tensor(10000.0, device=t.device))
        freqs = torch.exp(torch.linspace(0.0, 1.0, half, device=t.device) * freq_exp)
        phase = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.proj(emb)


def _segment_mean(x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    """노드 피처/좌표의 그래프별 평균을 계산한다.

    한 줄 요약:
        packed 노드 텐서를 그래프 인덱스로 분할(segment)하여 그래프별 평균을 낸다.

    생성이유:
        미니배치는 여러 그래프의 노드를 하나로 합쳐 담는다. 무게중심 계산,
        omega/v_lin 풀링, 레퍼런스/포켓 중심 계산 등에서 그래프별 평균이 반복
        필요하므로 공용 헬퍼로 분리했다.

    역할:
        같은 그래프에 속한 노드들의 평균 벡터를 그래프 수만큼 산출한다.

    메커니즘:
        index_add_로 그래프별 합과 노드 개수를 누적한 뒤 개수로 나눈다. 분모는
        clamp_min(1.0)로 0 나눗셈을 방지한다.

    Args:
        x (Tensor): 노드 텐서 ``[N, D]``.
        batch (Tensor): 노드별 그래프 인덱스 ``[N]``.
        num_graphs (int): 미니배치 내 그래프 수.

    Returns:
        Tensor: 그래프별 평균 텐서 ``[num_graphs, D]``.

    파이프라인 단계:
        3단계(모델 본체). forward와 각종 head/기저 계산에서 광범위하게 사용된다.
    """
    out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    return out / cnt.clamp_min(1.0)


def build_radius_edges(pos: Tensor, batch: Tensor, cutoff: float, max_neighbors: int) -> Tensor:
    """완전히 벡터화된 그래프 내 반경 에지 생성 (원자별 Python 루프 없음).

    한 줄 요약:
        동일 그래프 내에서 컷오프 거리 안의 가장 가까운 이웃들로 방향성 에지를
        구성한다.

    생성이유:
        EGNN/트랜스포머 블록은 이웃 메시지 패싱을 수행하므로 원자 그래프의 에지
        목록이 필요하다. 원자별 루프 없이 행렬 연산으로 빠르게 반경 그래프를
        만들기 위해 작성되었다.

    역할:
        쿼리 원자 좌표로부터 반경 이웃 그래프를 만들어 메시지 패싱의 연결성을
        제공한다.

    메커니즘:
        전체 쌍 거리 제곱을 계산하고, 같은 그래프이며 자기 자신이 아니고 컷오프
        이내인 쌍만 마스킹한다. max_neighbors가 설정되면 각 행에 대해 topk(가장
        가까운 것)만 남긴 뒤 nonzero로 에지 인덱스를 추출한다. 에지 ``(i, j)``는
        원자 ``j`` -> 원자 ``i``로의 메시지를 뜻한다.

    Args:
        pos (Tensor): 원자 좌표 ``[N, 3]``.
        batch (Tensor): 원자별 그래프 인덱스 ``[N]``.
        cutoff (float): 이웃으로 인정할 최대 거리.
        max_neighbors (int): 각 원자가 유지하는 최대 이웃 수(0 또는 N 이상이면 제한 없음).

    Returns:
        Tensor: 방향성 에지 인덱스 ``[2, E]`` (행 0=목적지 i, 행 1=출발 j).

    파이프라인 단계:
        3단계(모델 본체). forward에서 등변 블록 입력 그래프 구성에 사용된다.
    """
    n = pos.size(0)
    if n == 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)

    diff = pos[:, None, :] - pos[None, :, :]  # [n, n, 3]
    dist_sq = (diff * diff).sum(-1)  # [n, n]
    same = batch[:, None] == batch[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    mask = same & (~eye) & (dist_sq <= cutoff * cutoff)

    if 0 < max_neighbors < n:
        d = dist_sq.masked_fill(~mask, float("inf"))
        topk_idx = d.topk(min(max_neighbors, n), dim=1, largest=False).indices
        keep = torch.zeros_like(mask)
        keep.scatter_(1, topk_idx, True)
        mask = mask & keep

    i, j = mask.nonzero(as_tuple=True)
    return torch.stack([i, j], dim=0)

def build_cross_edges(
    lig_pos: Tensor,
    lig_batch: Tensor,
    pkt_pos: Tensor,
    pkt_batch: Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[Tensor, Tensor]:
    """컷오프 이내의 리간드->포켓 에지 생성 (동일 그래프), 완전히 벡터화됨.

    한 줄 요약:
        각 리간드(쿼리) 원자에 대해 컷오프 안의 가장 가까운 포켓 원자들로 향하는
        크로스 에지를 만든다.

    생성이유:
        포켓 컨디셔닝(PocketInteraction)은 리간드 원자와 인근 포켓 원자 사이의
        크로스 메시지 패싱을 수행한다. 이를 위해 리간드->포켓 연결성을 효율적으로
        구성해야 하므로 작성되었다.

    역할:
        리간드 원자와 같은 그래프에 속한 포켓 원자 사이의 근접 쌍을 추출한다.

    메커니즘:
        리간드-포켓 간 전체 쌍 거리 제곱을 계산하고, 같은 그래프이며 컷오프 이내인
        쌍만 마스킹한다. max_neighbors가 설정되면 리간드 원자별로 가장 가까운
        포켓 원자만 topk로 남긴 뒤 nonzero로 인덱스를 추출한다. 포켓/리간드가
        비어 있으면 빈 인덱스를 반환한다.

    Args:
        lig_pos (Tensor): 리간드(쿼리) 원자 좌표 ``[Nl, 3]``.
        lig_batch (Tensor): 리간드 원자별 그래프 인덱스 ``[Nl]``.
        pkt_pos (Tensor): 포켓 원자 좌표 ``[Npk, 3]``.
        pkt_batch (Tensor): 포켓 원자별 그래프 인덱스 ``[Npk]``.
        cutoff (float): 크로스 에지로 인정할 최대 거리.
        max_neighbors (int): 리간드 원자당 유지할 최대 포켓 이웃 수.

    Returns:
        tuple[Tensor, Tensor]: ``(lig_idx, pkt_idx)`` 형태의 에지 인덱스 쌍.
            각각 같은 길이의 1차원 텐서로 (리간드 원자, 포켓 원자) 쌍을 나타낸다.

    파이프라인 단계:
        3단계(모델 본체). use_pocket_conditioning 활성 시 forward에서 호출된다.
    """
    empty = torch.empty(0, dtype=torch.long, device=lig_pos.device)
    if pkt_pos is None or pkt_pos.numel() == 0 or lig_pos.numel() == 0:
        return empty, empty

    diff = lig_pos[:, None, :] - pkt_pos[None, :, :]  # [nl, npk, 3]
    dist_sq = (diff * diff).sum(-1)  # [nl, npk]
    same = lig_batch[:, None] == pkt_batch[None, :]
    mask = same & (dist_sq <= cutoff * cutoff)

    npk = pkt_pos.size(0)
    if 0 < max_neighbors < npk:
        d = dist_sq.masked_fill(~mask, float("inf"))
        topk_idx = d.topk(min(max_neighbors, npk), dim=1, largest=False).indices
        keep = torch.zeros_like(mask)
        keep.scatter_(1, topk_idx, True)
        mask = mask & keep

    lig_idx, pkt_idx = mask.nonzero(as_tuple=True)
    return lig_idx, pkt_idx


class PocketInteraction(nn.Module):
    """리간드 원자가 결합 부위의 형태를 감지할 수 있도록 하는 크로스 메시지 패싱.

    포켓은 고정된 컨텍스트이다 (원자가 절대 이동하지 않음). 각 리간드 원자에 대해
    (a) 인근 포켓 원자로부터 집계된 불변 피처 업데이트와,
    (b) 포켓 표면에 대한 방향 정보를 rigid head에 제공하는 등변 포켓 형태 벡터
    ``sum_j w_ij (p_j - x_i)`` (단순 중심이 아닌)를 생성한다.

    생성이유:
        단순히 포켓 무게중심만 조건으로 주면 결합 부위의 방향성/형태 정보를 잃는다.
        리간드 원자가 자신 주변의 포켓 표면 형태를 직접 감지하도록 크로스 어텐션
        형태의 메시지 패싱을 제공하기 위해 도입되었다.

    역할:
        리간드 원자별로 인근 포켓 원자의 정보를 모아 (a) 불변 피처 업데이트와
        (b) 등변 포켓 형태 벡터를 산출하여, 본체 피처와 rigid head 기저를 보강한다.

    주요 변수(속성):
        fallback_token: 포켓 원자 타입이 없을 때 포켓 노드 피처로 쓰이는 학습형 토큰.
        msg_mlp: 리간드/포켓 피처와 거리로부터 불변 메시지를 만드는 MLP.
        gate_mlp: 각 에지의 벡터 기여를 (0,1) 범위로 제한하는 시그모이드 게이트 MLP.
        norm: 집계된 피처 업데이트에 적용하는 LayerNorm.

    파이프라인 단계:
        3단계(모델 본체). use_pocket_conditioning 활성 시 forward에서 사용된다.
    """

    def __init__(self, hidden_dim: int) -> None:
        """포켓 크로스 인터랙션 모듈을 초기화한다.

        생성하는 주요 속성:
            fallback_token: 포켓 원자 타입 부재 시 사용할 폴백 노드 피처(학습 파라미터).
            msg_mlp: 불변 메시지 생성 MLP.
            gate_mlp: 벡터 기여 게이트(시그모이드 스칼라) MLP.
            norm: 피처 업데이트 정규화용 LayerNorm.

        Args:
            hidden_dim (int): 은닉 피처 차원.
        """
        super().__init__()
        # 포켓 원자 타입을 사용할 수 없을 때 포켓 노드 피처로 사용되는 폴백 토큰
        self.fallback_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_lig: Tensor, x_lig: Tensor, x_pkt: Tensor, h_pkt: Tensor, lig_idx: Tensor, pkt_idx: Tensor) -> tuple[Tensor, Tensor]:
        """``h_pkt``는 포켓 원자별 피처이다 (원자 타입 임베딩 또는 폴백 토큰).

        한 줄 요약:
            리간드-포켓 크로스 에지를 따라 메시지를 모아 리간드 피처 업데이트와
            등변 포켓 형태 벡터를 생성한다.

        생성이유:
            리간드 원자가 주변 결합 부위의 형태/방향을 인지하도록 포켓 정보를
            본체 피처와 rigid head 기저에 주입해야 한다.

        역할:
            각 리간드 원자에 대해 (a) 정규화된 불변 피처 업데이트와 (b) 게이트로
            범위가 제한된 등변 포켓 형태 벡터를 반환한다.

        메커니즘:
            에지별 상대 벡터 rij(리간드->포켓)와 거리 dij를 구하고, 리간드 피처·
            포켓 토큰·거리를 이어 msg_mlp/gate_mlp에 넣는다. 메시지는 index_add_로
            리간드 원자에 합산한 뒤 차수(deg)로 나눠 평균낸다. 벡터는 시그모이드
            게이트를 곱한 rij를 합산 후 평균낸다. 평균+게이트로 포켓 원자 수에
            무관하게 출력이 발산하지 않도록 한다(단순 합산은 omega 폭발->NaN 유발).
            에지가 없으면 0 텐서를 반환한다.

        Args:
            h_lig (Tensor): 리간드 원자 피처 ``[Nl, H]``.
            x_lig (Tensor): 리간드 원자 좌표 ``[Nl, 3]`` (무게중심 기준 정렬).
            x_pkt (Tensor): 포켓 원자 좌표 ``[Npk, 3]``.
            h_pkt (Tensor): 포켓 원자 피처 ``[Npk, H]`` (원자타입 임베딩 또는 폴백 토큰).
            lig_idx (Tensor): 크로스 에지의 리간드 원자 인덱스 ``[E]``.
            pkt_idx (Tensor): 크로스 에지의 포켓 원자 인덱스 ``[E]``.

        Returns:
            tuple[Tensor, Tensor]: ``(h_update, vec)``.
                h_update ``[Nl, H]``는 정규화된 불변 피처 업데이트,
                vec ``[Nl, 3]``는 등변 포켓 형태 벡터.

        파이프라인 단계:
            3단계(모델 본체). forward에서 포켓 컨디셔닝 시 호출된다.
        """
        n_lig = h_lig.size(0)
        h_update = torch.zeros_like(h_lig)
        vec = torch.zeros(n_lig, 3, device=h_lig.device, dtype=h_lig.dtype)
        if lig_idx.numel() == 0:
            return h_update, vec

        rij = x_pkt[pkt_idx] - x_lig[lig_idx]  # [E, 3], ligand -> pocket
        dij = safe_norm(rij, dim=-1, keepdim=True)  # [E, 1]
        tok = h_pkt[pkt_idx]
        feat = torch.cat([h_lig[lig_idx], tok, dij], dim=-1)

        # 평균 집계 + 시그모이드 게이트는 리간드 원자가 보는 포켓 원자 수에 관계없이
        # 두 출력을 유한 범위 내로 유지한다 (실제 포켓은 100~200개 원자를 가짐).
        # 단순 합산을 사용하면 포켓 형태 벡터가 발산 -> rigid 속도 폭발 -> NaN 발생.
        ones = torch.ones(lig_idx.size(0), 1, device=h_lig.device, dtype=h_lig.dtype)
        deg = torch.zeros(n_lig, 1, device=h_lig.device, dtype=h_lig.dtype)
        deg.index_add_(0, lig_idx, ones)
        deg = deg.clamp_min(1.0)
        
        h_update.index_add_(0, lig_idx, self.msg_mlp(feat))
        gate = torch.sigmoid(self.gate_mlp(feat))  # (0, 1): 각 에지의 벡터 기여를 범위 내로 제한
        vec.index_add_(0, lig_idx, gate * rij)     # |기여| <= cutoff
        return self.norm(h_update / deg), vec / deg

class EquivariantBlock(nn.Module):
    """단순 EGNN 스타일 블록: 스칼라 메시지 + 상대 벡터 집계.

    한 줄 요약:
        E(3)-등변 메시지 패싱 한 층으로, 노드 스칼라 피처와 좌표를 동시에
        업데이트하는 EGNN 블록.

    생성이유:
        정렬 벡터 필드는 회전/평행이동에 대해 등변이어야 한다. EGNN 구조는
        상대 벡터에 불변 스칼라 가중치를 곱해 좌표를 갱신하므로 등변성을 자연히
        만족한다. 이 등변 표현을 본체 백본으로 쌓기 위해 정의되었다.

    역할:
        에지를 따라 불변 메시지를 만들어 피처를 갱신하고, 상대 벡터를 스칼라
        가중 합산하여 좌표를 등변하게 갱신한다.

    주요 변수(속성):
        phi_e: 이웃 쌍 피처와 거리로부터 에지 메시지를 만드는 MLP.
        phi_h: 집계 메시지와 기존 피처로 노드 피처를 갱신하는 MLP.
        phi_x: 에지 메시지로부터 좌표 갱신 스칼라 가중치를 산출하는 MLP.

    파이프라인 단계:
        3단계(모델 본체). forward에서 num_blocks개가 순차 적용되는 백본 블록이다.
    """

    def __init__(self, hidden_dim: int) -> None:
        """EGNN 등변 블록을 초기화한다.

        생성하는 주요 속성:
            phi_e: 에지 메시지 MLP.
            phi_h: 노드 피처 갱신 MLP.
            phi_x: 좌표 갱신 가중치(스칼라) MLP.

        Args:
            hidden_dim (int): 은닉 피처 차원.
        """
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phi_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """등변 메시지 패싱 블록 한 번을 실행한다.

        한 줄 요약:
            에지 메시지로 노드 피처를 갱신하고 상대 벡터를 스칼라 가중 합산하여
            좌표를 등변하게 갱신한다.

        생성이유:
            등변성을 유지하면서 이웃 정보를 전파하는 한 층의 연산이 필요하다.

        역할:
            피처 h와 좌표 x를 한 단계 갱신하여 다음 블록에 넘긴다.

        메커니즘:
            각 에지 (i, j)에서 상대 벡터 rij=x_i-x_j와 거리 dij를 구한다
            (safe_norm으로 rij=0에서도 기울기 안정). phi_e로 에지 메시지 e_ij를
            만들고, phi_x(e_ij)를 스칼라 가중치로 rij에 곱해 좌표 업데이트 dx를
            i로 합산(등변). 피처는 e_ij를 i로 합산한 m과 함께 phi_h를 통과시켜
            잔차 갱신한다. 에지가 없으면 입력을 그대로 반환한다.

        Args:
            h (Tensor): 노드 스칼라 피처 ``[N, H]``.
            x (Tensor): 노드 좌표 ``[N, 3]``.
            edge_index (Tensor): 방향성 에지 ``[2, E]``.

        Returns:
            tuple[Tensor, Tensor]: 업데이트된 노드 피처 ``[N, H]``와 좌표 ``[N, 3]``.

        파이프라인 단계:
            3단계(모델 본체). forward의 백본 루프에서 호출된다.
        """
        if edge_index.numel() == 0:
            return h, x

        i, j = edge_index[0], edge_index[1]
        rij = x[i] - x[j]
        dij = safe_norm(rij, dim=-1, keepdim=True)  # safe_norm: rij == 0일 때도 유한한 기울기를 보장

        e_ij = self.phi_e(torch.cat([h[i], h[j], dij], dim=-1))

        # 좌표 업데이트 (등변): sum alpha_ij * (x_i - x_j)
        alpha_ij = self.phi_x(e_ij)
        dx_msg = alpha_ij * rij
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_msg)
        x = x + dx

        # 피처 업데이트: 노드 i로 메시지 집계
        m = torch.zeros_like(h)
        m.index_add_(0, i, e_ij)
        h = h + self.phi_h(torch.cat([h, m], dim=-1))
        return h, x


class TorchMDEquivariantTransformerBlock(nn.Module):
    """TorchMD-NET 스타일의 경량 등변 트랜스포머 블록.

    한 줄 요약:
        거리 바이어스를 포함한 멀티헤드 어텐션으로 피처를 갱신하고, 어텐션된
        스칼라 메시지로 좌표를 등변하게 갱신하는 트랜스포머 블록.

    생성이유:
        EGNN보다 표현력이 높은 어텐션 기반 등변 백본 대안을 제공하기 위해
        도입되었다(torchmd_et 백본 옵션).

    역할:
        쿼리/키/값 투영과 거리 의존 바이어스로 세그먼트 소프트맥스 어텐션을 수행해
        피처를 갱신하고, 어텐션 기반 게이트로 좌표를 갱신한다.

    주요 변수(속성):
        num_heads / head_dim: 어텐션 헤드 수와 헤드별 차원.
        q_proj/k_proj/v_proj/o_proj: 어텐션의 쿼리/키/값/출력 선형 투영.
        dist_proj: 에지 거리를 헤드별 어텐션 바이어스로 변환하는 MLP.
        coord_gate: 어텐션 메시지로부터 좌표 갱신 게이트를 산출하는 MLP.
        norm: 잔차 후 LayerNorm.
        ffn: 포지션와이즈 피드포워드 네트워크.

    파이프라인 단계:
        3단계(모델 본체). torchmd_et 백본 선택 시 등변 블록 대안으로 사용된다.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        """등변 트랜스포머 블록을 초기화한다.

        생성하는 주요 속성:
            num_heads/head_dim: 헤드 수와 헤드 차원(hidden_dim이 num_heads로
                나누어떨어져야 함).
            q_proj/k_proj/v_proj/o_proj: 어텐션 투영 레이어.
            dist_proj: 거리->헤드별 바이어스 MLP.
            coord_gate: 좌표 갱신 게이트 MLP.
            norm: LayerNorm.
            ffn: 피드포워드 네트워크.

        Args:
            hidden_dim (int): 은닉 피처 차원.
            num_heads (int): 어텐션 헤드 수(기본 4).

        Raises:
            ValueError: hidden_dim이 num_heads로 나누어떨어지지 않을 때.
        """
        super().__init__()
        self.num_heads = int(max(1, num_heads))
        self.head_dim = hidden_dim // self.num_heads
        if self.head_dim == 0 or self.head_dim * self.num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads for torchmd_et backbone.")

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.num_heads))
        self.coord_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """등변 트랜스포머 블록 한 번을 실행한다.

        한 줄 요약:
            거리 바이어스 멀티헤드 어텐션으로 피처를 갱신하고, 어텐션 게이트로
            정규화된 상대 방향을 더해 좌표를 등변하게 갱신한다.

        생성이유:
            어텐션 가중치로 이웃 중요도를 학습해 EGNN보다 풍부한 메시지 패싱을
            수행하기 위함이다.

        역할:
            노드 피처 h와 좌표 x를 한 단계 갱신한다.

        메커니즘:
            에지별 상대 벡터 rij와 거리 dij를 구하고 q/k/v를 헤드별로 투영한다.
            (q_i·k_j)/sqrt(head_dim)에 거리 바이어스를 더해 어텐션 로짓을 만든 뒤,
            각 목적지 노드별 수신 에지에 대해 세그먼트 소프트맥스를 적용한다.
            가중된 v를 i로 합산하여 o_proj 잔차로 피처를 갱신하고 ffn+norm을 적용.
            좌표는 coord_gate(집계 메시지)를 정규화된 방향 rij/dij에 곱해 i로
            합산하여 갱신한다. 에지가 없으면 입력을 그대로 반환한다.

        Args:
            h (Tensor): 노드 스칼라 피처 ``[N, H]``.
            x (Tensor): 노드 좌표 ``[N, 3]``.
            edge_index (Tensor): 방향성 에지 ``[2, E]``.

        Returns:
            tuple[Tensor, Tensor]: 업데이트된 피처 ``[N, H]``와 좌표 ``[N, 3]``.

        파이프라인 단계:
            3단계(모델 본체). torchmd_et 백본의 블록 루프에서 호출된다.
        """
        if edge_index.numel() == 0:
            return h, x

        i, j = edge_index[0], edge_index[1]
        rij = x[i] - x[j]
        dij = torch.norm(rij, dim=-1, keepdim=True).clamp_min(1e-8)

        q = self.q_proj(h).view(h.size(0), self.num_heads, self.head_dim)
        k = self.k_proj(h).view(h.size(0), self.num_heads, self.head_dim)
        v = self.v_proj(h).view(h.size(0), self.num_heads, self.head_dim)

        q_i = q[i]
        k_j = k[j]
        v_j = v[j]
        dist_bias = self.dist_proj(dij)  # [E, heads]
        attn_logits = (q_i * k_j).sum(dim=-1) / (self.head_dim**0.5) + dist_bias

        # 각 목적지 노드 i에 대한 수신 에지의 세그먼트 소프트맥스.
        attn = torch.zeros_like(attn_logits)
        n_nodes = h.size(0)
        for node in range(n_nodes):
            mask = i == node
            if mask.any():
                attn[mask] = torch.softmax(attn_logits[mask], dim=0)

        msg = attn.unsqueeze(-1) * v_j
        agg = torch.zeros(h.size(0), self.num_heads, self.head_dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, i, msg)
        agg = agg.reshape(h.size(0), -1)

        h = h + self.o_proj(agg)
        h = self.norm(h + self.ffn(h))

        # 어텐션된 스칼라 메시지로부터의 등변 좌표 업데이트.
        gate = self.coord_gate(agg[i])
        dx_msg = gate * (rij / dij)
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_msg)
        x = x + dx
        return h, x


class ETFlowAlignModel(nn.Module):
    """쿼리 원자를 위한 간결한 E(3)-등변 시간 의존 벡터 필드 모델.

    한 줄 요약:
        AlignmentBatch와 시간 t를 입력받아 쿼리 원자별 속도장(velocity field)을
        예측하는 SE(3) flow matching 본체 모델.

    생성이유:
        ETFlowAlign의 핵심은 query 리간드를 reference/pocket 조건에 맞춰 정렬하는
        것이며, 이를 위해 조건과 시간에 의존하는 등변 속도장 v_theta(x_t, t, cond)가
        필요하다. 이 모델은 등변 백본과 다양한 출력 head를 조합해 그 속도장을
        구현한다.

    역할:
        쿼리 좌표를 무게중심 정렬한 뒤 원자타입/시간/레퍼런스 방향으로 임베딩하고,
        (옵션) 포켓 컨디셔닝을 거쳐 등변 블록을 통과시킨 다음 선택된 head로
        원자별 속도 벡터를 산출한다.

    주요 변수(속성):
        edge_cutoff / max_neighbors: 쿼리 그래프 반경 에지 구성 파라미터.
        atom_embed: 원자 타입 임베딩.
        use_atom_index_embed/use_direct_vector_head/use_equivariant_basis_head/
            use_rigid_head/use_pocket_conditioning: 각 옵션 활성 플래그.
        pocket_cutoff/pocket_max_neighbors: 포켓 크로스 에지 파라미터.
        max_atoms: 원자 인덱스 임베딩 상한.
        pocket_interaction: 포켓 컨디셔닝 모듈(옵션).
        _n_rigid_bases: rigid head의 기저 개수(포켓 사용 시 5, 아니면 4).
        atom_index_embed: 로컬 원자 인덱스 임베딩(옵션).
        time_embed: 시간 임베딩 모듈.
        in_proj: 원자/시간/레퍼런스 컨디셔닝을 은닉 차원으로 합치는 입력 투영.
        blocks: 등변 메시지 패싱 블록 리스트(백본).
        out_gate: 레거시 제약적 등변 head.
        out_vec: 디버그용 직접 벡터 head(옵션, 비등변).
        out_basis_coeff: 등변 기저 조합 head 계수 MLP(옵션).
        out_rigid_omega/out_rigid_vlin: rigid head의 각/선형 속도 계수 MLP(옵션).

    파이프라인 단계:
        3단계(모델 본체). 학습(train_step), flow matching, 샘플링(sampler) 전반에서
        속도장 예측기로 사용된다.
    """

    def __init__(
        self,
        atom_vocab_size: int = 128,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_blocks: int = 4,
        edge_cutoff: float = 6.0,
        max_neighbors: int = 32,
        use_atom_index_embed: bool = False,
        use_direct_vector_head: bool = False,
        use_equivariant_basis_head: bool = False,
        use_rigid_head: bool = False,
        use_pocket_conditioning: bool = False,
        pocket_cutoff: float = 8.0,
        pocket_max_neighbors: int = 16,
        use_node_attr: bool = False,
        node_attr_dim: int = 5,
        max_atoms: int = 256,
    ) -> None:
        """ETFlowAlign 모델을 구성하고 모든 서브모듈/head를 초기화한다.

        생성하는 주요 속성:
            atom_embed/time_embed/in_proj/blocks: 입력 임베딩과 등변 백본.
            out_gate(레거시), out_vec(디버그), out_basis_coeff(등변 기저),
            out_rigid_omega/out_rigid_vlin(rigid): 옵션에 따라 생성되는 출력 head.
            pocket_interaction/atom_index_embed: 포켓·원자인덱스 옵션 모듈.
            _n_rigid_bases: 포켓 사용 여부에 따른 rigid 기저 수.

        메커니즘:
            각 use_* 플래그를 bool로 저장하고 해당 옵션에 필요한 서브모듈만
            조건부로 생성한다. in_proj 입력 차원은 원자(hidden) + 시간(time_embed)
            + 레퍼런스 컨디셔닝(방향 3 + 거리 1)을 합친 값이다.

        Args:
            atom_vocab_size (int): 원자 타입 임베딩 어휘 크기.
            hidden_dim (int): 은닉 피처 차원.
            time_embed_dim (int): 시간 임베딩 차원.
            num_blocks (int): 등변 백본 블록 수.
            edge_cutoff (float): 쿼리 그래프 반경 에지 컷오프.
            max_neighbors (int): 쿼리 원자당 최대 이웃 수.
            use_atom_index_embed (bool): 로컬 원자 인덱스 임베딩 사용 여부(디버그).
            use_direct_vector_head (bool): 직접 벡터 head 사용 여부(비등변 디버그).
            use_equivariant_basis_head (bool): 등변 기저 조합 head 사용 여부.
            use_rigid_head (bool): 강체 속도 필드 head 사용 여부(direction A).
            use_pocket_conditioning (bool): 포켓 컨디셔닝 사용 여부.
            pocket_cutoff (float): 포켓 크로스 에지 컷오프.
            pocket_max_neighbors (int): 리간드 원자당 최대 포켓 이웃 수.
            use_node_attr (bool): 노드 속성 사용 플래그(현재 시그니처 보존용).
            node_attr_dim (int): 노드 속성 차원.
            max_atoms (int): 로컬 원자 인덱스 임베딩 상한.

        파이프라인 단계:
            3단계(모델 본체). 모델 인스턴스 생성 시(학습/추론 시작) 호출된다.
        """
        super().__init__()
        self.edge_cutoff = float(edge_cutoff)
        self.max_neighbors = int(max_neighbors)

        self.atom_embed = nn.Embedding(atom_vocab_size, hidden_dim)
        self.use_atom_index_embed = bool(use_atom_index_embed)
        self.use_direct_vector_head = bool(use_direct_vector_head)
        self.use_equivariant_basis_head = bool(use_equivariant_basis_head)
        self.use_rigid_head = bool(use_rigid_head)
        self.use_pocket_conditioning = bool(use_pocket_conditioning)
        self.pocket_cutoff = float(pocket_cutoff)
        self.pocket_max_neighbors = int(pocket_max_neighbors)
        self.max_atoms = int(max_atoms)

        if self.use_pocket_conditioning:
            self.pocket_interaction = PocketInteraction(hidden_dim)

        # 포켓 컨디셔닝이 활성화되면 rigid head는 5번째 기저(포켓 형태 벡터)를 추가로 가짐
        self._n_rigid_bases = 5 if self.use_pocket_conditioning else 4
        
        if self.use_atom_index_embed:
            self.atom_index_embed = nn.Embedding(self.max_atoms, hidden_dim)

        self.time_embed = SimpleTimeEmbedding(time_embed_dim)
        self.in_proj = nn.Linear(hidden_dim + time_embed_dim + 4, hidden_dim)

        self.blocks = nn.ModuleList([EquivariantBlock(hidden_dim) for _ in range(num_blocks)])

        # 기존 프로덕션 헤드:
        #   v_i = alpha_i x_i
        # E(3)-등변이지만 많은 정렬 플로우에 대해 너무 제약적이다.
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 디버그 헤드:
        #   v_i = MLP(h_i)
        # 표현력이 높고 과적합 검증에 유용하지만 E(3)-등변이 아니다.
        if self.use_direct_vector_head:
            self.out_vec = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )

        # 프로덕션 호환 표현력 있는 헤드:
        #   v_i = sum_k a_{ik} b_{ik}
        # 여기서 b_{ik}는 등변 벡터 기저이고 a_{ik}는 불변 스칼라이다.
        if self.use_equivariant_basis_head:
            self.out_basis_coeff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )
        # Rigid (direction A) 헤드:
        #   v_i = omega_g x (x_i - c_g) + v_lin_g
        # 그래프별 강체 속도 필드 (분자당 6 자유도). 필드가 구조적으로 강체 운동이므로,
        # 이를 적분해도 분자 내 거리가 전혀 변하지 않는다: 원본 형태의 결합/각도 구조가
        # 정확히 보존된다. omega_g와 v_lin_g는 등변 벡터 기저의 그래프 풀링 조합으로
        # 구성되므로, 필드는 입력과 함께 회전한다 (`in_proj`의 레퍼런스 방향 컨디셔닝까지
        # E(3)-등변).
        if self.use_rigid_head:
            self.out_rigid_omega = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self._n_rigid_bases),
            )
            self.out_rigid_vlin = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self._n_rigid_bases),
            )
            
    def _reference_context(self, batch: AlignmentBatch) -> tuple[Tensor, Tensor]:
        """쿼리 원자로부터 레퍼런스 중심까지의 방향과 거리를 반환한다.

        한 줄 요약:
            각 쿼리 원자에서 레퍼런스 무게중심으로 향하는 단위 방향과 거리를
            컨디셔닝 피처로 산출한다.

        생성이유:
            모델이 "쿼리를 어디로 정렬해야 하는지"를 알려면 레퍼런스 위치에 대한
            기하 단서가 입력 피처에 포함되어야 한다.

        역할:
            in_proj 입력에 결합될 레퍼런스 방향(3차원)과 거리(1차원) 컨디셔닝을
            제공한다.

        메커니즘:
            그래프별 레퍼런스 무게중심을 _segment_mean으로 구하고, 쿼리 원자별로
            (중심-원자) 델타를 계산해 거리와 정규화 방향으로 분리한다. 레퍼런스가
            없으면 0 벡터/0 거리를 반환한다.

        Args:
            batch (AlignmentBatch): 정렬 미니배치.

        Returns:
            tuple[Tensor, Tensor]: ``(direction, dist)``.
                direction ``[Nq, 3]``는 단위 방향, dist ``[Nq, 1]``는 거리.

        파이프라인 단계:
            3단계(모델 본체). forward의 입력 컨디셔닝 구성에 사용된다.
        """
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            zero = torch.zeros_like(batch.query_pos)
            return zero, torch.zeros(batch.query_pos.size(0), 1, device=batch.query_pos.device, dtype=batch.query_pos.dtype)

        num_graphs = int(batch.query_batch.max().item()) + 1
        ref_center = _segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        delta = ref_center[batch.query_batch] - batch.query_pos
        dist = safe_norm(delta, dim=-1, keepdim=True)
        direction = delta / dist.clamp_min(1e-8)
        return direction, dist

    def _reference_delta_basis(self, batch: AlignmentBatch) -> Tensor:
        """레퍼런스 중심에서 쿼리 원자까지의 벡터 기저를 반환한다.

        기저:
            b_ref,i = c_ref[graph(i)] - q_i

        이 벡터는 평행 이동 불변이며 회전 등변이다.

        한 줄 요약:
            레퍼런스 무게중심에서 각 쿼리 원자로 향하는 등변 벡터 기저를 만든다.

        생성이유:
            등변 head(basis/rigid)가 속도를 표현하려면 입력과 함께 회전하는 벡터
            기저가 필요하다. 레퍼런스 방향은 그 핵심 기저 중 하나다.

        역할:
            head들이 불변 스칼라 계수와 곱해 등변 속도를 구성하도록 벡터 기저를
            공급한다.

        메커니즘:
            그래프별 레퍼런스 무게중심을 _segment_mean으로 계산한 뒤 쿼리 원자별로
            (중심 - 원자)를 반환한다. 레퍼런스가 없으면 0 벡터를 반환한다.

        Args:
            batch (AlignmentBatch): 정렬 미니배치.

        Returns:
            Tensor: 레퍼런스 델타 기저 ``[Nq, 3]``.

        파이프라인 단계:
            3단계(모델 본체). _equivariant_basis_head/_rigid_head에서 사용된다.
        """
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        ref_center = _segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        return ref_center[batch.query_batch] - batch.query_pos

    def _pocket_delta_basis(self, batch: AlignmentBatch) -> Tensor:
        """포켓 중심에서 쿼리 원자까지의 벡터 기저를 반환한다.

        기저:
            b_pocket,i = c_pocket[graph(i)] - q_i

        포켓 좌표가 없으면 영 벡터를 반환한다.

        한 줄 요약:
            포켓 무게중심에서 각 쿼리 원자로 향하는 등변 벡터 기저를 만든다.

        생성이유:
            결합 부위(포켓) 방향 정보를 등변 head의 기저로 제공해 포켓을 향한
            정렬을 표현하기 위함이다.

        역할:
            head들이 곱할 등변 포켓 방향 기저를 공급한다.

        메커니즘:
            그래프별 포켓 무게중심을 _segment_mean으로 계산한 뒤 쿼리 원자별로
            (중심 - 원자)를 반환한다. 포켓 좌표가 없으면 0 벡터를 반환한다.

        Args:
            batch (AlignmentBatch): 정렬 미니배치.

        Returns:
            Tensor: 포켓 델타 기저 ``[Nq, 3]``.

        파이프라인 단계:
            3단계(모델 본체). _equivariant_basis_head/_rigid_head에서 사용된다.
        """
        if batch.pocket_pos is None or batch.pocket_batch is None or batch.pocket_batch.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        pocket_center = _segment_mean(batch.pocket_pos, batch.pocket_batch, num_graphs)
        return pocket_center[batch.query_batch] - batch.query_pos

    def _neighbor_vector_basis(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """쿼리-쿼리 이웃 집계 벡터 기저를 반환한다 (이웃에 대한 평균).

        기저:
            b_nbr,i = mean_{j in N(i)} (x_j - x_i)

        합산이 아닌 평균을 사용하면 이웃 수에 관계없이 크기가 유한하게 유지된다;
        경계 없는 합산은 실제 데이터에서 omega 폭발 -> NaN/거대한 손실의 원인이었다.
        평행 이동 불변이며 회전 등변이다.

        한 줄 요약:
            각 쿼리 원자에 대해 이웃으로 향하는 상대 벡터의 평균을 등변 기저로
            만든다.

        생성이유:
            분자 내부 국소 기하(이웃 배치) 정보를 등변 head 기저로 제공하기 위함이다.
            평균을 사용해 이웃 수에 따른 크기 발산(NaN)을 방지한다.

        역할:
            head들이 곱할 국소 이웃 집계 등변 기저를 공급한다.

        메커니즘:
            에지 (i, j)에서 rij=x_j-x_i를 i로 index_add_ 합산하고, 각 i의 이웃 수
            (deg)로 나눠 평균낸다. 에지가 없으면 0 벡터를 반환한다.

        Args:
            x (Tensor): 쿼리 원자 좌표 ``[Nq, 3]`` (보통 중심화된 입력 좌표).
            edge_index (Tensor): 쿼리-쿼리 방향성 에지 ``[2, E]``.

        Returns:
            Tensor: 이웃 집계 기저 ``[Nq, 3]``.

        파이프라인 단계:
            3단계(모델 본체). _equivariant_basis_head/_rigid_head에서 사용된다.
        """
        if edge_index.numel() == 0:
            return torch.zeros_like(x)

        i, j = edge_index[0], edge_index[1]
        rij = x[j] - x[i]
        out = torch.zeros_like(x)
        out.index_add_(0, i, rij)
        deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, i, torch.ones(i.size(0), 1, device=x.device, dtype=x.dtype))
        return out / deg.clamp_min(1.0)

    def _local_atom_index(self, query_batch: Tensor) -> Tensor:
        """선택적 디버그 원자 인덱스 임베딩을 위한 그래프 내 로컬 원자 인덱스를 반환한다.

        한 줄 요약:
            각 그래프 내부에서 0,1,2,... 로 매겨지는 로컬 원자 인덱스를 만든다.

        생성이유:
            디버그용 원자 인덱스 임베딩(과적합 검증 등)을 사용할 때, 그래프마다
            독립적인 원자 순번이 필요하다.

        역할:
            use_atom_index_embed 활성 시 atom_index_embed의 입력으로 쓸 로컬
            인덱스를 제공한다.

        메커니즘:
            그래프 인덱스의 고유값을 순회하며 각 그래프 마스크 내부에 arange(n)을
            채워 넣는다.

        Args:
            query_batch (Tensor): 쿼리 원자별 그래프 인덱스 ``[Nq]``.

        Returns:
            Tensor: 그래프 내 로컬 원자 인덱스 ``[Nq]``.

        파이프라인 단계:
            3단계(모델 본체). forward에서 원자 인덱스 임베딩 옵션 시 사용된다.
        """
        local_index = torch.zeros_like(query_batch)
        for g in query_batch.unique(sorted=True):
            mask = query_batch == g
            n = int(mask.sum().item())
            local_index[mask] = torch.arange(n, device=query_batch.device, dtype=query_batch.dtype)
        return local_index

    def _equivariant_basis_head(self, h: Tensor, x: Tensor, batch: AlignmentBatch, edge_index: Tensor) -> Tensor:
        """등변 벡터 기저를 조합하여 속도를 예측한다.

        스칼라 계수는 스칼라 노드 피처로부터 예측되므로 불변이다. 기저는 입력 좌표와
        함께 회전한다. 따라서 가중 합산은 E(3)-등변이다.

        기저:
            0. x: 등변 블록 이후 쿼리 중심 좌표 기저.
            1. reference delta: 레퍼런스 중심 - 쿼리 원자.
            2. pocket delta: 포켓 중심 - 쿼리 원자.
            3. neighbor aggregate: sum_j (x_j - x_i).

        한 줄 요약:
            불변 스칼라 계수와 등변 벡터 기저들의 가중 합으로 원자별 속도를
            예측하는 등변 head.

        생성이유:
            레거시 out_gate head는 표현력이 부족하다. 프로덕션 호환을 유지하면서도
            여러 등변 기저를 조합해 더 풍부한 속도장을 표현하기 위해 도입되었다.

        역할:
            원자별 속도 벡터를 등변하게 산출한다.

        메커니즘:
            out_basis_coeff(h)로 불변 스칼라 계수 [N,4]를 만들고, 쿼리 좌표·
            레퍼런스/포켓 델타·이웃 집계의 4개 등변 기저를 [N,4,3]으로 쌓아
            계수와 곱한 뒤 기저 축으로 합산한다. 계수는 불변, 기저는 등변이므로
            결과는 E(3)-등변이다.

        Args:
            h (Tensor): 노드 스칼라 피처 ``[N, H]``.
            x (Tensor): 등변 블록 이후 좌표 ``[N, 3]``.
            batch (AlignmentBatch): 정렬 미니배치.
            edge_index (Tensor): 쿼리-쿼리 에지 ``[2, E]``.

        Returns:
            Tensor: 원자별 속도 벡터 ``[N, 3]``.

        파이프라인 단계:
            3단계(모델 본체). use_equivariant_basis_head 활성 시 forward에서 호출.
        """
        coeff = self.out_basis_coeff(h)  # [N, 4]

        basis_query = x
        basis_reference = self._reference_delta_basis(batch)
        basis_pocket = self._pocket_delta_basis(batch)
        basis_neighbor = self._neighbor_vector_basis(x, edge_index)

        bases = torch.stack(
            [
                basis_query,
                basis_reference,
                basis_pocket,
                basis_neighbor,
            ],
            dim=1,
        )  # [N, 4, 3]

        return (coeff.unsqueeze(-1) * bases).sum(dim=1)
    def _rigid_head(
        self,
        h: Tensor,
        x: Tensor,
        x_in: Tensor,
        batch: AlignmentBatch,
        edge_index: Tensor,
        pocket_basis: Optional[Tensor] = None,
    ) -> Tensor:
        """그래프별 강체 속도 필드를 예측한다 (direction A).

        단계:
            1. 원자별 등변 벡터 기저를 구성한다 (basis head와 동일한 집합).
            2. 불변 스칼라 가중 기저 합산으로 원자별 omega/v_lin을 구성한다.
            3. 그래프당 하나의 omega_g와 v_lin_g로 풀링한다 (평균이 등변성을 보존).
            4. 중심화된 입력 좌표에 대한 강체 필드로 확장한다:
                   v_i = omega_g x (x_in_i - c_g) + v_lin_g

        ``x_in``은 중심화된 *입력* 쿼리 좌표이다 (블록 업데이트된 ``x``가 아님).
        따라서 반환된 필드는 실제 샘플러 상태의 정확한 강체 운동이다. 등변 기저도
        ``x_in`` (및 레퍼런스/포켓 기하)로부터 구성된다: 블록 업데이트된 ``x``는
        정규화되지 않아 블록을 거치며 커질 수 있고, 이로 인해 ``omega``가 폭발하고
        손실이 불안정해진다. ``h``는 여전히 불변 계수로서 학습된 메시지 패싱
        컨텍스트를 담고 있다.

        한 줄 요약:
            그래프당 하나의 각속도/선형속도로 정의되는 강체 속도 필드를 예측해
            분자 내부 기하를 정확히 보존한다.

        생성이유:
            정렬 과정에서 결합 길이/각도 등 분자 내부 구조가 변형되면 안 된다.
            구조적으로 강체 운동만 표현하는 head를 두면 적분해도 내부 거리가
            전혀 변하지 않아 원본 형태가 정확히 보존된다.

        역할:
            그래프별 강체 속도 필드를 산출하여 forward의 rigid 경로 출력으로 쓴다.

        메커니즘:
            등변 기저(쿼리 좌표·레퍼런스/포켓 델타·이웃 집계, 포켓 사용 시 포켓
            형태 벡터 추가)를 스칼라 계수와 곱해 원자별 omega/v_lin을 만든 뒤,
            그래프 평균(_segment_mean)으로 그래프당 omega_g/v_lin_g로 풀링한다.
            이를 중심화 좌표에 대해 v_i = omega_g x (x_in_i - c_g) + v_lin_g 로
            확장한다.

        Args:
            h (Tensor): 노드 스칼라 피처 ``[N, H]`` (불변 계수 소스).
            x (Tensor): 등변 블록 이후 좌표 ``[N, 3]`` (여기선 직접 사용하지 않음).
            x_in (Tensor): 중심화된 입력 쿼리 좌표 ``[N, 3]``.
            batch (AlignmentBatch): 정렬 미니배치.
            edge_index (Tensor): 쿼리-쿼리 에지 ``[2, E]``.
            pocket_basis (Optional[Tensor]): 포켓 형태 벡터 ``[N, 3]`` (옵션).

        Returns:
            Tensor: 원자별 강체 속도 벡터 ``[N, 3]``.

        파이프라인 단계:
            3단계(모델 본체). use_rigid_head 활성 시 forward에서 호출된다.
        """
        num_graphs = int(batch.query_batch.max().item()) + 1

        basis_list = [
            x_in,                                       # 쿼리 중심 좌표
            self._reference_delta_basis(batch),         # 레퍼런스 무게중심 -> 쿼리
            self._pocket_delta_basis(batch),            # 포켓 무게중심 -> 쿼리
            self._neighbor_vector_basis(x_in, edge_index),  # 쿼리 이웃 집계
        ]
        if self.use_pocket_conditioning:
            # 포켓 형태 벡터 (단순 중심이 아닌 방향성 표면 정보)
            basis_list.append(pocket_basis if pocket_basis is not None else torch.zeros_like(x_in))
        bases = torch.stack(basis_list, dim=1)  # [N, n_bases, 3]

        omega_atom = (self.out_rigid_omega(h).unsqueeze(-1) * bases).sum(dim=1)  # [N, 3]
        vlin_atom = (self.out_rigid_vlin(h).unsqueeze(-1) * bases).sum(dim=1)  # [N, 3]

        omega = _segment_mean(omega_atom, batch.query_batch, num_graphs)  # [G, 3]
        vlin = _segment_mean(vlin_atom, batch.query_batch, num_graphs)  # [G, 3]

        com = _segment_mean(x_in, batch.query_batch, num_graphs)  # [G, 3]
        rel = x_in - com[batch.query_batch]  # [N, 3]
        omega_i = omega[batch.query_batch]  # [N, 3]
        vlin_i = vlin[batch.query_batch]  # [N, 3]

        return torch.cross(omega_i, rel, dim=-1) + vlin_i
    
    def forward(self, batch: AlignmentBatch, t_graph: Tensor) -> Tensor:
        """쿼리 속도 필드 ``v_theta(x_t, t, cond)``를 예측한다.

        한 줄 요약:
            AlignmentBatch와 시간 t로부터 쿼리 원자별 속도 벡터를 예측하는 모델의
            메인 진입점.

        생성이유:
            flow matching 학습/샘플링은 현재 상태 x_t와 시간 t에서의 속도장을
            반복적으로 요구한다. 이 메서드가 임베딩->컨디셔닝->백본->head의 전체
            예측을 한 번에 수행한다.

        역할:
            입력을 무게중심 정렬·임베딩하고, (옵션) 포켓 컨디셔닝과 등변 백본을
            거쳐 선택된 head로 속도 벡터를 산출한다.

        메커니즘:
            1) 쿼리 좌표를 그래프별 무게중심 기준으로 중심화(x, x_in).
            2) 원자타입 임베딩(+옵션 원자인덱스 임베딩), 시간 임베딩, 레퍼런스
               방향/거리를 이어 in_proj로 은닉 피처 h를 만든다.
            3) 포켓 컨디셔닝 활성 시 포켓 좌표를 같은 중심으로 정렬하고
               build_cross_edges + pocket_interaction으로 h를 보강, pocket_basis 획득.
            4) build_radius_edges로 쿼리 그래프를 만들고 등변 블록들을 순차 적용.
            5) head 우선순위에 따라 출력: direct(디버그) > rigid > equivariant basis
               > 레거시 gate(v=gate*x). 빈 입력이면 0을 반환한다.

        Args:
            batch (AlignmentBatch): 정렬 미니배치.
            t_graph (Tensor): 그래프당 시간 ``[B]``.

        Returns:
            Tensor: 쿼리 원자의 속도 벡터 ``[Nq, 3]``.

        파이프라인 단계:
            3단계(모델 본체). 4단계(flow matching 손실), 5단계(학습 루프),
            7단계(ODE 샘플링)에서 매 스텝 호출된다.
        """
        if batch.query_pos.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        com = _segment_mean(batch.query_pos, batch.query_batch, num_graphs)
        x = batch.query_pos - com[batch.query_batch]
        x_in = x.clone()  # 중심화된 입력 좌표, rigid head에서 사용됨
        
        h_atom = self.atom_embed(batch.query_atom_type)

        if self.use_atom_index_embed:
            local_index = self._local_atom_index(batch.query_batch)
            if int(local_index.max().item()) >= self.max_atoms:
                raise ValueError(f"local atom index exceeds max_atoms={self.max_atoms}; increase --max-atoms.")
            h_atom = h_atom + self.atom_index_embed(local_index)

        h_t = self.time_embed(t_graph)[batch.query_batch]
        ref_dir, ref_dist = self._reference_context(batch)

        h = self.in_proj(torch.cat([h_atom, h_t, ref_dir, ref_dist], dim=-1))

        # 포켓 형태 컨디셔닝: 리간드 피처를 보강하고 고정된 결합 부위 원자로부터
        # 방향성 포켓 형태 기저를 구성한다 (리간드 무게중심 기준 좌표계에서).
        pocket_basis = None
        if (
            self.use_pocket_conditioning
            and batch.pocket_pos is not None
            and batch.pocket_batch is not None
            and batch.pocket_batch.numel() > 0
        ):
            xp = batch.pocket_pos - com[batch.pocket_batch]
            if batch.pocket_atom_type is not None:
                h_pkt = self.atom_embed(batch.pocket_atom_type)  # 포켓 화학 정보 (공유 원자 임베딩)
            else:
                h_pkt = self.pocket_interaction.fallback_token.unsqueeze(0).expand(xp.size(0), -1)
            lig_idx, pkt_idx = build_cross_edges(
                x_in, batch.query_batch, xp, batch.pocket_batch,
                self.pocket_cutoff, self.pocket_max_neighbors,
            )
            h_pocket, pocket_basis = self.pocket_interaction(h, x_in, xp, h_pkt, lig_idx, pkt_idx)
            h = h + h_pocket
        
        edge_index = build_radius_edges(x, batch.query_batch, self.edge_cutoff, self.max_neighbors)

        for block in self.blocks:
            h, x = block(h, x, edge_index)

        # 최우선: 디버그용 직접 헤드.
        # 의도적으로 비등변이며 정상 동작 확인 용도로만 사용해야 한다.
        if self.use_direct_vector_head:
            return self.out_vec(h)

        # Direction A: 강체 속도 필드 (분자 내 기하 구조 정확히 보존).
        if self.use_rigid_head:
            return self._rigid_head(h, x, x_in, batch, edge_index, pocket_basis)

        # 프로덕션 호환 표현력 있는 등변 헤드.
        if self.use_equivariant_basis_head:
            return self._equivariant_basis_head(h, x, batch, edge_index)

        # 레거시 제약적 등변 헤드.
        gate = self.out_gate(h)
        v = gate * x
        return v