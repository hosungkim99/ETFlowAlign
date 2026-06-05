"""ETFlowAlign의 플로우 매칭 목적함수 및 정렬 특화 확률 경로."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel
from .validation import validate_alignment_batch

def _rotation_to_axis_angle(rotation: Tensor) -> Tensor:
    """열 우선(column-convention) 회전 행렬 ``[3,3]``을 축-각도 벡터 ``[3]``으로 변환한다.

    반환되는 벡터 ``omega``는 방향이 회전축이고 크기가 회전각이므로,
    ``exp(skew(omega)) == rotation``이 성립한다.
    """
    cos = ((rotation.diagonal().sum() - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.arccos(cos)
    sin = torch.sin(theta)

    if float(theta) < 1e-6:
        return torch.zeros(3, device=rotation.device, dtype=rotation.dtype)

    if float(sin.abs()) < 1e-6:
        # theta가 pi에 가까울 때: (R + I)/2 = k k^T; 가장 큰 대각 원소로 축을 복원한다.
        a = (rotation + torch.eye(3, device=rotation.device, dtype=rotation.dtype)) * 0.5
        diag = a.diagonal().clamp_min(0.0)
        i = int(torch.argmax(diag))
        axis = a[i] / torch.sqrt(diag[i]).clamp_min(1e-8)
        axis = axis / axis.norm().clamp_min(1e-8)
        return axis * theta

    w = torch.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return (w / (2.0 * sin)) * theta


def _axis_angle_to_matrix(omega: Tensor) -> Tensor:
    """로드리게스 공식: 축-각도 벡터 ``[3]`` -> 회전 행렬 ``[3,3]`` (열 우선 표기법)."""
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    theta = omega.norm()
    if float(theta) < 1e-8:
        return eye
    k = omega / theta
    skew = torch.zeros(3, 3, device=omega.device, dtype=omega.dtype)
    skew[0, 1], skew[0, 2] = -k[2], k[1]
    skew[1, 0], skew[1, 2] = k[2], -k[0]
    skew[2, 0], skew[2, 1] = -k[1], k[0]
    return eye + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)

@dataclass
class FlowMatchingConfig:
    sigma: float = 0.05
    source_type: Literal[
    "gaussian",
    "reference_anchored",
    "query_perturbed",
    "input_query",
] = "reference_anchored"
    source_noise_scale: float = 0.5
    time_eps: float = 1e-4
    use_kabsch_alignment: bool = True
    harmonic_prior_strength: float = 0.0
    center_source: bool = True
    center_target: bool = True
    fixed_t: float | None = None
    path_type: Literal["linear", "rigid"] = "linear"


class AlignmentFlowMatcher:
    def __init__(self, config: FlowMatchingConfig) -> None:
        if config.path_type == "rigid" and config.source_type != "input_query":
            raise ValueError(
                "path_type='rigid' requires source_type='input_query': the rigid path and "
                "rigid-source inference both use query_pos as the source conformer. "
                f"Got source_type={config.source_type!r}."
            )
        self.config = config

    def sample_time(self, num_graphs: int, device: torch.device) -> Tensor:
        if self.config.fixed_t is not None:
            fixed_t = float(self.config.fixed_t)
            if not 0.0 <= fixed_t <= 1.0:
                raise ValueError(f"fixed_t must be in [0, 1], got {fixed_t}")
            return torch.full((num_graphs,), fixed_t, device=device)
        return torch.empty(num_graphs, device=device).uniform_(
            self.config.time_eps,
            1.0 - self.config.time_eps,
        )
    def sigma_t(self, t_node: Tensor) -> Tensor:
        return self.config.sigma * torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))

    def sigma_dot_t(self, t_node: Tensor) -> Tensor:
        denom = torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))
        return self.config.sigma * 0.5 * (1.0 - 2.0 * t_node) / denom

    def sample_source(self, batch: AlignmentBatch, target_query_pos: Tensor | None = None) -> Tensor:
        stype = self.config.source_type

        if stype == "gaussian":
            return torch.randn_like(batch.query_pos)
        
        if stype == "input_query":
            return batch.query_pos.clone()

        if stype == "query_perturbed":
            if target_query_pos is None:
                raise ValueError("source_type='query_perturbed' requires target_query_pos.")
            return target_query_pos + self.config.source_noise_scale * torch.randn_like(target_query_pos)

        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            return torch.randn_like(batch.query_pos)
        num_graphs = int(batch.reference_batch.max().item()) + 1
        center = torch.zeros(num_graphs, 3, device=batch.reference_pos.device, dtype=batch.reference_pos.dtype)
        count = torch.zeros(num_graphs, 1, device=batch.reference_pos.device, dtype=batch.reference_pos.dtype)
        center.index_add_(0, batch.reference_batch, batch.reference_pos)
        count.index_add_(
            0,
            batch.reference_batch,
            torch.ones_like(batch.reference_batch, dtype=batch.reference_pos.dtype).unsqueeze(-1),
        )
        center = center / count.clamp_min(1.0)
        return center[batch.query_batch] + self.config.source_noise_scale * torch.randn_like(batch.query_pos)

    def _kabsch_align_source_to_target(self, x0: Tensor, target_query_pos: Tensor, batch_index: Tensor) -> Tensor:
        out = x0.clone()
        for g in batch_index.unique(sorted=True):
            mask = batch_index == g
            p = x0[mask]
            q = target_query_pos[mask]
            if p.size(0) < 2:
                out[mask] = q
                continue
            pc = p - p.mean(0, keepdim=True)
            qc = q - q.mean(0, keepdim=True)
            h = pc.transpose(0, 1) @ qc
            u, _, vT = torch.linalg.svd(h)
            r = vT.transpose(0, 1) @ u.transpose(0, 1)
            if torch.det(r) < 0:
                vT[-1, :] *= -1
                r = vT.transpose(0, 1) @ u.transpose(0, 1)
            aligned = pc @ r + q.mean(0, keepdim=True)
            out[mask] = aligned
        return out

    def _apply_harmonic_prior_if_needed(self, x0: Tensor, batch: AlignmentBatch) -> Tensor:
        if self.config.harmonic_prior_strength <= 0:
            return x0
        center = torch.zeros(int(batch.query_batch.max().item()) + 1, 3, device=x0.device, dtype=x0.dtype)
        count = torch.zeros(center.size(0), 1, device=x0.device, dtype=x0.dtype)
        center.index_add_(0, batch.query_batch, x0)
        count.index_add_(0, batch.query_batch, torch.ones(x0.size(0), 1, device=x0.device, dtype=x0.dtype))
        center = center / count.clamp_min(1.0)
        return x0 - self.config.harmonic_prior_strength * (x0 - center[batch.query_batch])
    
    def _center_by_graph(self, x: Tensor, batch_index: Tensor) -> Tensor:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_graphs == 0:
            return x

        mean = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        count = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)

        mean.index_add_(0, batch_index, x)
        count.index_add_(
            0,
            batch_index,
            torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype),
        )

        mean = mean / count.clamp_min(1.0)
        return x - mean[batch_index]
    
    def _build_rigid_training_state(
        self,
        batch: AlignmentBatch,
        x0: Tensor,
        x1: Tensor,
        t_node: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """소스 포즈 ``x0``와 타겟 포즈 ``x1`` 사이의 SE(3) 지오데식 경로.

        각 그래프에서 소스와 타겟은 서로 다른 강체 포즈를 가진 동일한 컨포머이므로,
        최적 강체 사상(회전 ``R``, 질량중심 이동)은 Kabsch 알고리즘으로 복원된다.
        경로는 SO(3) 지오데식을 따라 회전을 보간하고 질량중심은 선형으로 보간한다::

            x_t = (x0 - c0) @ R(t).T + ((1 - t) c0 + t c1)
            u_t = omega x (x_t - c(t)) + (c1 - c0)

        여기서 ``R(t) = exp(t * skew(omega))``이다. 모든 중간 ``x_t``는 소스 컨포머의
        강체 변환이므로 분자 내부 기하구조가 정확히 보존되고, ``u_t``는 순수 강체 속도장
        (강체 헤드가 매칭)이다.
        """
        x_t = torch.empty_like(x1)
        u_t = torch.empty_like(x1)

        for g in batch.query_batch.unique(sorted=True):
            mask = batch.query_batch == g
            p = x0[mask]
            q = x1[mask]
            t = t_node[mask][0]

            c0 = p.mean(0, keepdim=True)
            c1 = q.mean(0, keepdim=True)
            c_t = (1.0 - t) * c0 + t * c1
            pc = p - c0

            if p.size(0) < 2:
                x_t[mask] = pc + c_t
                u_t[mask] = (c1 - c0).expand_as(p)
                continue

            qc = q - c1
            h = pc.transpose(0, 1) @ qc
            u_svd, _, vT = torch.linalg.svd(h)
            r = vT.transpose(0, 1) @ u_svd.transpose(0, 1)
            if torch.det(r) < 0:
                vT = vT.clone()
                vT[-1, :] *= -1
                r = vT.transpose(0, 1) @ u_svd.transpose(0, 1)

            # Kabsch ``r``은 qc ~= pc @ r (행 우선)을 만족하며, 열 우선 회전은 그 전치이다.
            # h = pc.T @ qc일 때, Kabsch ``r = V @ U.T``는 소스를 타겟으로 매핑하는
            # 열 우선 회전이다 (r @ pc_col ~= qc_col); 행 포인트에 적용: pc @ r.T.
            omega = _rotation_to_axis_angle(r)
            if not torch.isfinite(omega).all():
                # 퇴화된 기하구조(거의 공선/평면 리간드, 각도 ~= pi)는 축-각도 추출을
                # 불안정하게 만든다. 하나의 불량 복합체가 배치 전체를 NaN으로 만들지 않도록
                # 이 그래프에 대해 이동만 하는 경로로 폴백한다.
                x_t[mask] = pc + c_t
                u_t[mask] = (c1 - c0).expand_as(p)
                continue
            r_t = _axis_angle_to_matrix(t * omega)

            xt = pc @ r_t.transpose(0, 1) + c_t
            rel = xt - c_t
            x_t[mask] = xt
            u_t[mask] = torch.cross(omega.unsqueeze(0).expand_as(rel), rel, dim=-1) + (c1 - c0)

        # 최종 안전망: 위의 그래프별 가드가 알려진 퇴화 케이스를 처리하지만,
        # 남아 있는 비정상 값(큰 배치에서 불량 복합체 하나)을 여기서 0으로 만들어
        # 배치 전체의 손실을 NaN으로 만들지 않는다.
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        u_t = torch.nan_to_num(u_t, nan=0.0, posinf=0.0, neginf=0.0)
        return x_t, u_t
    
    def build_training_state(self, batch: AlignmentBatch, target_query_pos: Tensor, t_graph: Tensor) -> tuple[Tensor, Tensor]:
        validate_alignment_batch(batch, target_query_pos=target_query_pos, require_target=True)
        t_node = t_graph[batch.query_batch]


        if self.config.path_type == "rigid":
            # rigid 경로는 입력 쿼리 컨포머를 소스로 직접 사용한다: 소스->타겟 맵이
            # 강체 운동이 되려면 유효한 컨포머여야 한다. reference_anchored / gaussian
            # 소스는 무작위 포인트 클라우드가 되어 (비결정적이기도 하므로) sample_source를
            # 의도적으로 우회한다. 경로는 그래프별로 중심화하고 자체 SE(3) 지오데식을
            # 유도하므로, 소스 중심화 / 하모닉 프라이어 / Kabsch 사전 정렬 / 보간 노이즈는
            # 모두 불필요하거나 강체성을 깨뜨리므로 건너뛴다.
            return self._build_rigid_training_state(
                batch=batch, x0=batch.query_pos, x1=target_query_pos, t_node=t_node,
            )
        x0 = self.sample_source(batch=batch, target_query_pos=target_query_pos)
        if self.config.center_source:
            x0 = self._center_by_graph(x0, batch.query_batch)
        
        x1 = target_query_pos
        if self.config.center_target:
            x1 = self._center_by_graph(x1, batch.query_batch)
        x0 = self._apply_harmonic_prior_if_needed(x0=x0, batch=batch)
        if self.config.use_kabsch_alignment:
            x0 = self._kabsch_align_source_to_target(x0=x0, target_query_pos=x1, batch_index=batch.query_batch)
        eps = torch.randn_like(x1)
        sigma = self.sigma_t(t_node).unsqueeze(-1)
        sigma_dot = self.sigma_dot_t(t_node).unsqueeze(-1)
        x_t = (1.0 - t_node).unsqueeze(-1) * x0 + t_node.unsqueeze(-1) * x1 + sigma * eps
        u_t = (x1 - x0) + sigma_dot * eps
        return x_t, u_t

    def loss(self, pred_v: Tensor, target_u: Tensor, batch_index: Tensor) -> Tensor:
        per_atom = ((pred_v - target_u) ** 2).sum(dim=-1)
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_graphs == 0:
            return per_atom.mean() * 0.0
        graph_sum = torch.zeros(num_graphs, device=per_atom.device, dtype=per_atom.dtype)
        graph_cnt = torch.zeros(num_graphs, device=per_atom.device, dtype=per_atom.dtype)
        graph_sum.index_add_(0, batch_index, per_atom)
        graph_cnt.index_add_(0, batch_index, torch.ones_like(per_atom))
        return (graph_sum / graph_cnt.clamp_min(1.0)).mean()

def endpoint_bond_length_loss(
    pred_v: Tensor,
    x_t: Tensor,
    t_graph: Tensor,
    batch: AlignmentBatch,
) -> Tensor:
    """한 스텝 추정 엔드포인트 x1_hat에 대한 결합 길이 정규화."""
    if batch.query_bond_index is None or batch.query_bond_length is None:
        return pred_v.sum() * 0.0

    if batch.query_bond_index.numel() == 0:
        return pred_v.sum() * 0.0

    bond_index = batch.query_bond_index.to(device=x_t.device)
    target_length = batch.query_bond_length.to(device=x_t.device, dtype=x_t.dtype)

    i, j = bond_index[0], bond_index[1]
    t_node = t_graph[batch.query_batch].unsqueeze(-1)

    x1_hat = x_t + (1.0 - t_node) * pred_v
    pred_length = torch.linalg.norm(x1_hat[i] - x1_hat[j], dim=-1)

    return ((pred_length - target_length) ** 2).mean()

def flow_matching_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    batch: AlignmentBatch,
    target_query_pos: Tensor,
    lambda_bond: float = 0.0,
    ) -> Tensor:
    validate_alignment_batch(batch, target_query_pos=target_query_pos, require_target=True)
    num_graphs = int(batch.query_batch.max().item()) + 1
    t_graph = matcher.sample_time(num_graphs=num_graphs, device=batch.query_pos.device)
    x_t, u_t = matcher.build_training_state(batch=batch, target_query_pos=target_query_pos, t_graph=t_graph)
    step_batch = AlignmentBatch(
        query_pos=x_t,
        query_atom_type=batch.query_atom_type,
        query_batch=batch.query_batch,
        reference_pos=batch.reference_pos,
        reference_atom_type=batch.reference_atom_type,
        reference_batch=batch.reference_batch,
        pocket_pos=batch.pocket_pos,
        pocket_batch=batch.pocket_batch,
        pocket_atom_type=batch.pocket_atom_type,
        query_node_attr=batch.query_node_attr,
        reference_node_attr=batch.reference_node_attr,
        query_bond_index=batch.query_bond_index,
        query_bond_length=batch.query_bond_length,
    )
    pred_v = model(step_batch, t_graph=t_graph)
    fm_loss = matcher.loss(pred_v=pred_v, target_u=u_t, batch_index=batch.query_batch)

    if lambda_bond <= 0.0:
        return fm_loss

    bond_loss = endpoint_bond_length_loss(
        pred_v=pred_v,
        x_t=x_t,
        t_graph=t_graph,
        batch=batch,
    )

    return fm_loss + float(lambda_bond) * bond_loss