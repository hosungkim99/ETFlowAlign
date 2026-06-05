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

    한 줄 요약:
        SO(3) 회전 행렬을 로그 사상하여 축-각도(axis-angle) 표현으로 바꾼다.

    생성이유:
        rigid 경로(``_build_rigid_training_state``)에서 SE(3) 지오데식을 따라
        회전을 시간 보간하려면, Kabsch가 복원한 회전 행렬 ``R``을 각속도 벡터
        ``omega``로 변환해야 한다. ``omega``가 있어야 ``R(t)=exp(t*skew(omega))``의
        중간 회전과 강체 속도장 ``omega x (x_t - c_t)``를 계산할 수 있다.

    역할:
        회전 행렬을 받아, 회전축 방향이고 크기가 회전각인 벡터 ``omega``를 반환한다.
        이는 회전 행렬의 행렬 로그(matrix logarithm)에 대응한다.

    메커니즘(핵심 연산/알고리즘):
        - 회전각: ``theta = arccos((trace(R)-1)/2)`` 로 복원한다(코사인은 ``[-1,1]``로 클램프).
        - 일반 케이스: 반대칭 성분 ``(R - R^T)``에서 회전축을 추출해 ``theta``로 스케일한다.
        - 퇴화 케이스 1(``theta`` ~= 0): 영벡터를 반환한다.
        - 퇴화 케이스 2(``theta`` ~= pi, ``sin`` ~= 0): ``(R+I)/2 = k k^T`` 관계를 이용해
          가장 큰 대각 원소가 있는 행에서 축을 복원하고 정규화한다.

    파라미터:
        rotation (Tensor): 형상 ``[3,3]``의 SO(3) 회전 행렬(열 우선 표기).

    반환:
        Tensor: 형상 ``[3]``의 축-각도 벡터 ``omega`` (방향=축, 크기=각도).

    파이프라인 단계:
        4단계(학습 목표/확률경로) - rigid 경로의 SE(3) 지오데식 구성을 위한 보조 변환.
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
    """로드리게스 공식: 축-각도 벡터 ``[3]`` -> 회전 행렬 ``[3,3]`` (열 우선 표기법).

    한 줄 요약:
        축-각도 벡터를 지수 사상(로드리게스 공식)으로 SO(3) 회전 행렬로 변환한다.

    생성이유:
        rigid 경로에서 시간 ``t``에 비례하는 중간 회전 ``R(t)=exp(t*skew(omega))``를
        만들어야 SE(3) 지오데식 위의 중간 포즈 ``x_t``를 생성할 수 있다.
        ``_rotation_to_axis_angle``의 역변환에 해당한다.

    역할:
        회전축/각도 정보를 담은 벡터 ``omega``를 받아 대응하는 회전 행렬을 반환한다.

    메커니즘(핵심 연산/알고리즘):
        - 회전각 ``theta = ||omega||`` 가 거의 0이면 단위 행렬을 반환한다.
        - 단위 축 ``k = omega/theta`` 의 반대칭 행렬 ``skew(k)`` 를 구성한다.
        - 로드리게스 공식 ``I + sin(theta)*skew + (1-cos(theta))*skew@skew`` 로 회전 행렬을 만든다.

    파라미터:
        omega (Tensor): 형상 ``[3]``의 축-각도 벡터(방향=축, 크기=각도).

    반환:
        Tensor: 형상 ``[3,3]``의 SO(3) 회전 행렬(열 우선 표기).

    파이프라인 단계:
        4단계(학습 목표/확률경로) - rigid 경로의 SE(3) 지오데식 회전 보간을 위한 보조 변환.
    """
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
    """플로우 매칭 학습의 확률경로와 소스 분포를 제어하는 설정.

    한 줄 요약:
        ``AlignmentFlowMatcher``가 사용할 소스 분포·확률경로·노이즈·정렬 옵션을
        모아 둔 dataclass 설정.

    생성이유:
        flow matching 학습에서 소스(x0) 분포 선택, 확률경로 종류(linear/rigid),
        보간 노이즈 스케일, 사전 정렬(Kabsch) 사용 여부 등 다수의 하이퍼파라미터를
        하나의 구조체로 묶어 실험·재현·검증을 용이하게 하기 위함이다.

    역할:
        학습 목표(x_t, u_t) 생성에 필요한 모든 설정값을 보관하고
        ``AlignmentFlowMatcher``에 전달한다.

    정의되는 주요 변수(dataclass 필드):
        sigma (float): 확률경로의 보간 노이즈 진폭 기준값. ``sigma_t``가 시간에 따라 변조한다.
        source_type (Literal): 소스 포즈 x0를 만드는 방식.
            "gaussian"=표준정규, "reference_anchored"=레퍼런스 질량중심에 노이즈,
            "query_perturbed"=타겟 쿼리에 노이즈, "input_query"=입력 쿼리 그대로(rigid 필수).
        source_noise_scale (float): 소스 생성 시 더하는 가우시안 노이즈 배율.
        time_eps (float): 시간 t 샘플링 구간의 경계 여유값(0,1 양 끝 회피).
        use_kabsch_alignment (bool): 소스를 타겟에 Kabsch로 사전 강체 정렬할지 여부(linear 경로).
        harmonic_prior_strength (float): 소스를 그래프 질량중심 쪽으로 수축시키는 하모닉 프라이어 강도.
        center_source (bool): 소스를 그래프별 질량중심으로 중심화할지 여부.
        center_target (bool): 타겟을 그래프별 질량중심으로 중심화할지 여부.
        fixed_t (float | None): 지정 시 시간 t를 무작위 대신 고정값으로 사용(디버깅/특정 시점 학습).
        path_type (Literal): 확률경로 종류. "linear"=선형 보간+노이즈, "rigid"=SE(3) 지오데식.
    """

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
    """정렬 특화 플로우 매칭의 확률경로와 학습 목표를 생성하는 핵심 클래스.

    한 줄 요약:
        소스 x0와 타겟 x1을 잇는 확률경로 위의 중간 상태 ``x_t``와 목표 속도 ``u_t``를
        생성하는 플로우 매칭 매처.

    생성이유:
        SE(3) 정렬 학습에서 모델이 학습할 회귀 목표(목표 속도장)를 정의해야 한다.
        선형 확률경로뿐 아니라 분자 내부 기하구조를 보존하는 rigid(SE(3) 지오데식)
        경로까지 일관된 인터페이스로 제공하기 위해 별도 클래스로 분리했다.

    역할:
        - 시간 t 샘플링(``sample_time``)과 시간 의존 노이즈 스케일(``sigma_t``, ``sigma_dot_t``) 제공.
        - 소스 포즈 생성(``sample_source``)과 사전 정렬·중심화·하모닉 프라이어 보조 처리.
        - 확률경로별 학습 상태 ``(x_t, u_t)`` 구성(``build_training_state``).
        - 예측 속도와 목표 속도의 그래프 평균 손실 계산(``loss``).

    정의되는 주요 변수(속성):
        config (FlowMatchingConfig): 소스 분포·경로·노이즈·정렬 옵션을 담은 설정 객체.

    파이프라인 단계:
        4단계(학습 목표/확률경로) - 학습 손실 계산의 목표를 만든다.
    """

    def __init__(self, config: FlowMatchingConfig) -> None:
        """설정을 검증하고 매처를 초기화한다.

        한 줄 요약:
            ``FlowMatchingConfig``를 받아 경로/소스 조합의 정합성을 검증한 뒤 저장한다.

        생성이유:
            rigid 경로는 소스가 유효한 컨포머(입력 쿼리)여야 강체 운동이 성립하므로,
            잘못된 조합을 학습 시작 전에 차단하기 위함이다.

        역할:
            설정의 ``path_type``/``source_type`` 조합을 검증하고 ``self.config``에 보관한다.

        메커니즘(핵심 연산/알고리즘):
            ``path_type == "rigid"`` 인데 ``source_type != "input_query"`` 이면
            ``ValueError``를 발생시켜 비강체 소스로 인한 잘못된 강체 경로 생성을 막는다.

        파라미터:
            config (FlowMatchingConfig): 플로우 매칭 확률경로/소스/노이즈 설정.

        생성 속성:
            self.config (FlowMatchingConfig): 이후 모든 메서드가 참조하는 설정 객체.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 매처 초기화 및 설정 검증.
        """
        if config.path_type == "rigid" and config.source_type != "input_query":
            raise ValueError(
                "path_type='rigid' requires source_type='input_query': the rigid path and "
                "rigid-source inference both use query_pos as the source conformer. "
                f"Got source_type={config.source_type!r}."
            )
        self.config = config

    def sample_time(self, num_graphs: int, device: torch.device) -> Tensor:
        """학습 스텝마다 그래프별 시간 t를 샘플링한다.

        한 줄 요약:
            확률경로 상의 시점 ``t in (0,1)`` 를 그래프 수만큼 샘플링한다.

        생성이유:
            플로우 매칭은 경로 위 임의의 시점에서 목표 속도를 회귀하므로,
            매 스텝 무작위(또는 고정) 시간을 뽑아 경로 전체를 균등하게 학습해야 한다.

        역할:
            ``fixed_t`` 설정 시 고정 시간 텐서를, 아니면 ``[time_eps, 1-time_eps]`` 균등분포에서
            샘플링한 시간 텐서를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            - ``fixed_t`` 가 지정되면 ``[0,1]`` 범위 검증 후 모든 그래프에 같은 값을 채운다.
            - 아니면 경계 0,1을 ``time_eps``만큼 회피한 균등분포에서 추출한다(노이즈 스케일 발산 방지).

        파라미터:
            num_graphs (int): 배치 내 그래프(분자) 수. 반환 텐서의 길이.
            device (torch.device): 결과 텐서를 둘 디바이스.

        반환:
            Tensor: 형상 ``[num_graphs]``의 그래프별 시간 값.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 확률경로 시점 샘플링.
        """
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
        """시간 의존 보간 노이즈의 표준편차 ``sigma(t)`` 를 계산한다.

        한 줄 요약:
            선형 확률경로에 더하는 노이즈의 시간별 진폭 ``sigma*sqrt(t(1-t))`` 를 반환한다.

        생성이유:
            소스/타겟 양 끝점(t=0,1)에서는 노이즈가 0이 되어 정확히 끝점을 통과하고,
            중간에서 최대가 되는 브리지 형태의 확률경로를 만들기 위함이다.

        역할:
            노드(원자)별 시간을 받아 그 시점의 노이즈 표준편차를 계산한다.

        메커니즘(핵심 연산/알고리즘):
            ``sigma * sqrt( clamp_min(t*(1-t), 1e-8) )`` 로 계산하며,
            클램프로 ``sqrt`` 내부가 음수/0이 되는 수치 문제를 방지한다.

        파라미터:
            t_node (Tensor): 형상 ``[N]``의 노드별 시간 값.

        반환:
            Tensor: 형상 ``[N]``의 노드별 노이즈 표준편차 ``sigma(t)``.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 선형 경로 ``x_t`` 의 노이즈 항 계산.
        """
        return self.config.sigma * torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))

    def sigma_dot_t(self, t_node: Tensor) -> Tensor:
        """노이즈 표준편차의 시간 미분 ``d sigma(t)/dt`` 를 계산한다.

        한 줄 요약:
            ``sigma_t``의 시간 도함수로, 목표 속도 ``u_t``의 노이즈 항 계수를 제공한다.

        생성이유:
            확률경로 ``x_t``에 시간 의존 노이즈가 들어가므로, 그에 정합하는 목표 속도
            ``u_t = (x1-x0) + sigma_dot(t)*eps`` 를 만들려면 노이즈의 시간 변화율이 필요하다.

        역할:
            노드별 시간을 받아 노이즈 표준편차의 순간 변화율을 반환한다.

        메커니즘(핵심 연산/알고리즘):
            ``sigma(t)=sigma*sqrt(t(1-t))`` 를 미분한
            ``sigma * 0.5 * (1-2t) / sqrt(clamp_min(t(1-t),1e-8))`` 를 계산한다.

        파라미터:
            t_node (Tensor): 형상 ``[N]``의 노드별 시간 값.

        반환:
            Tensor: 형상 ``[N]``의 노드별 노이즈 표준편차 시간 도함수.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 선형 경로 목표 속도 ``u_t`` 의 노이즈 항 계산.
        """
        denom = torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))
        return self.config.sigma * 0.5 * (1.0 - 2.0 * t_node) / denom

    def sample_source(self, batch: AlignmentBatch, target_query_pos: Tensor | None = None) -> Tensor:
        """확률경로의 시작점인 소스 포즈 ``x0`` 를 설정에 따라 생성한다.

        한 줄 요약:
            ``source_type`` 설정에 맞춰 소스 좌표 ``x0`` 를 만들어 반환한다.

        생성이유:
            플로우 매칭은 소스 분포에서 타겟 분포로의 경로를 학습하므로,
            학습 난이도/사전지식 활용도를 조절할 수 있는 다양한 소스 초기화가 필요하다.

        역할:
            네 가지 소스 유형(가우시안/입력쿼리/타겟섭동/레퍼런스앵커) 중 하나로
            쿼리 원자 좌표와 같은 형상의 소스 포즈를 생성한다.

        메커니즘(핵심 연산/알고리즘):
            - "gaussian": 쿼리와 같은 형상의 표준정규 노이즈.
            - "input_query": 입력 쿼리 좌표 복사(원본 보존).
            - "query_perturbed": 타겟 쿼리에 가우시안 노이즈를 더함(``target_query_pos`` 필수).
            - "reference_anchored"(기본): 레퍼런스가 없으면 가우시안으로 폴백하고,
              있으면 ``index_add_`` 로 그래프별 레퍼런스 질량중심을 구한 뒤
              쿼리 노드에 브로드캐스트하고 노이즈를 더한다.

        파라미터:
            batch (AlignmentBatch): 쿼리/레퍼런스 좌표·배치 인덱스를 담은 입력 배치.
            target_query_pos (Tensor | None): 타겟 쿼리 좌표. "query_perturbed"에서 필수.

        반환:
            Tensor: 형상 ``[N,3]``(쿼리 원자 수)의 소스 포즈 ``x0``.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 확률경로 시작점 x0 생성.
        """
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
        """그래프별로 소스 ``x0`` 를 타겟에 Kabsch 강체 정렬한다.

        한 줄 요약:
            각 분자에 대해 소스를 타겟에 최적 강체(회전+이동) 정렬해 학습 경로를 단축한다.

        생성이유:
            소스와 타겟의 전역 강체 자유도 차이를 미리 제거하면, 모델이 회전·이동
            정렬보다 미세 구조 보정에 집중하도록 하여 선형 경로의 학습 난이도를 낮춘다.

        역할:
            그래프 단위로 Kabsch 알고리즘을 적용해 정렬된 소스 좌표를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            그래프 마스크별로 소스 p, 타겟 q를 중심화한 뒤 공분산 ``h = pc^T qc`` 의
            SVD로 최적 회전 ``r``을 구한다. ``det(r)<0`` 이면 반사 보정을 적용한다.
            원자가 2개 미만이면 회전을 정의할 수 없어 소스를 타겟으로 직접 대체한다.
            최종적으로 ``pc @ r + q의 질량중심`` 으로 정렬한다.

        파라미터:
            x0 (Tensor): 형상 ``[N,3]``의 소스 포즈.
            target_query_pos (Tensor): 형상 ``[N,3]``의 타겟 포즈(정렬 기준).
            batch_index (Tensor): 형상 ``[N]``의 그래프 소속 인덱스.

        반환:
            Tensor: 형상 ``[N,3]``의 타겟에 정렬된 소스 포즈.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 선형 경로 소스 사전 정렬(전처리).
        """
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
        """필요 시 소스를 그래프 질량중심 쪽으로 수축시키는 하모닉 프라이어를 적용한다.

        한 줄 요약:
            ``harmonic_prior_strength`` 가 양수일 때 소스를 그래프 중심 쪽으로 끌어당긴다.

        생성이유:
            과도하게 흩어진 소스 포즈를 질량중심 쪽으로 모아 초기 분포를 더 응집된
            형태로 만들고, 학습 경로의 안정성을 높이기 위함이다.

        역할:
            설정 강도에 비례해 각 원자를 자기 그래프의 질량중심 방향으로 이동시킨다.

        메커니즘(핵심 연산/알고리즘):
            강도가 0 이하이면 그대로 반환한다. 아니면 ``index_add_`` 로 그래프별 질량중심을
            구한 뒤 ``x0 - strength*(x0 - center)`` 로 중심 쪽 선형 보간을 수행한다.

        파라미터:
            x0 (Tensor): 형상 ``[N,3]``의 소스 포즈.
            batch (AlignmentBatch): 그래프 소속 인덱스(``query_batch``)를 제공하는 배치.

        반환:
            Tensor: 형상 ``[N,3]``의 (필요 시) 수축된 소스 포즈.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 선형 경로 소스 전처리(선택적 프라이어).
        """
        if self.config.harmonic_prior_strength <= 0:
            return x0
        center = torch.zeros(int(batch.query_batch.max().item()) + 1, 3, device=x0.device, dtype=x0.dtype)
        count = torch.zeros(center.size(0), 1, device=x0.device, dtype=x0.dtype)
        center.index_add_(0, batch.query_batch, x0)
        count.index_add_(0, batch.query_batch, torch.ones(x0.size(0), 1, device=x0.device, dtype=x0.dtype))
        center = center / count.clamp_min(1.0)
        return x0 - self.config.harmonic_prior_strength * (x0 - center[batch.query_batch])
    
    def _center_by_graph(self, x: Tensor, batch_index: Tensor) -> Tensor:
        """좌표를 그래프별 질량중심으로 평행이동(중심화)한다.

        한 줄 요약:
            각 분자의 질량중심을 원점으로 옮겨 전역 이동(translation) 자유도를 제거한다.

        생성이유:
            소스/타겟을 동일 기준(질량중심 원점)으로 정렬해야 확률경로의 이동 성분이
            제거되고, 속도장 회귀가 회전·구조 변화에 집중되어 학습이 안정된다.

        역할:
            그래프 단위 평균을 빼서 중심화된 좌표를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            ``index_add_`` 로 그래프별 좌표 합과 개수를 누적해 평균을 구하고,
            각 원자에서 자기 그래프 평균을 뺀다. 그래프가 없으면 입력을 그대로 반환한다.

        파라미터:
            x (Tensor): 형상 ``[N,D]``의 좌표 텐서.
            batch_index (Tensor): 형상 ``[N]``의 그래프 소속 인덱스.

        반환:
            Tensor: 형상 ``[N,D]``의 그래프별 중심화된 좌표.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 소스/타겟 중심화 전처리.
        """
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

        한 줄 요약:
            소스->타겟 강체 운동을 SE(3) 지오데식으로 보간해 ``(x_t, u_t)`` 를 만든다.

        생성이유:
            선형 보간은 중간 포즈에서 분자 내부 결합 구조를 왜곡한다. rigid 경로는
            모든 중간 상태를 소스 컨포머의 강체 변환으로 유지하여 기하구조를 정확히
            보존하면서 정렬을 학습하기 위해 도입되었다.

        역할:
            그래프별로 Kabsch 회전을 구해 SO(3) 지오데식 회전 보간 + 선형 질량중심 보간으로
            중간 포즈 ``x_t`` 와 그에 정합하는 강체 목표 속도 ``u_t`` 를 생성한다.

        메커니즘(핵심 연산/알고리즘):
            - 그래프별 소스/타겟 질량중심 c0,c1과 중심화 좌표 pc,qc를 구한다.
            - 공분산 ``h=pc^T qc`` SVD로 회전 ``r``(필요 시 반사 보정)을 얻는다.
            - ``omega = log(r)`` (축-각도), 중간 회전 ``R(t)=exp(t*omega)`` 적용.
            - ``x_t = pc @ R(t)^T + c_t``, ``u_t = omega x (x_t - c_t) + (c1-c0)``.
            - 원자<2개 또는 비유한 ``omega`` 인 퇴화 케이스는 이동만 하는 경로로 폴백.
            - 마지막에 ``nan_to_num`` 안전망으로 배치 전체 NaN 전파를 차단한다.

        파라미터:
            batch (AlignmentBatch): 그래프 인덱스(``query_batch``)를 제공하는 배치.
            x0 (Tensor): 형상 ``[N,3]``의 소스 컨포머(입력 쿼리) 좌표.
            x1 (Tensor): 형상 ``[N,3]``의 타겟 컨포머 좌표.
            t_node (Tensor): 형상 ``[N]``의 노드별 시간 값.

        반환:
            tuple[Tensor, Tensor]: ``(x_t, u_t)``. 각각 형상 ``[N,3]``의 중간 포즈와
            강체 목표 속도장.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - rigid(SE(3) 지오데식) 경로의 학습 상태 생성.
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
        """확률경로 종류에 따라 학습 상태 ``(x_t, u_t)`` 를 구성하는 진입점.

        한 줄 요약:
            설정된 ``path_type`` 에 맞춰 중간 상태 ``x_t`` 와 목표 속도 ``u_t`` 를 만든다.

        생성이유:
            학습 손실 계산에 필요한 입력(모델에 넣을 ``x_t``)과 회귀 목표(``u_t``)를
            경로 종류와 무관하게 단일 호출로 얻기 위한 통합 인터페이스가 필요하다.

        역할:
            배치 유효성을 검증하고, rigid/linear 경로 중 해당 분기를 실행해
            ``(x_t, u_t)`` 를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            - 그래프별 시간 ``t_graph`` 를 노드 시간 ``t_node`` 로 확장한다.
            - rigid: 입력 쿼리를 소스로 ``_build_rigid_training_state`` 호출(중심화/Kabsch/노이즈 생략).
            - linear: 소스 샘플링 -> (선택) 소스 중심화 -> 타겟 중심화 -> (선택) 하모닉 프라이어
              -> (선택) Kabsch 정렬 후, 노이즈 ``eps`` 와 ``sigma_t``/``sigma_dot_t`` 로
              ``x_t=(1-t)x0+t x1+sigma*eps``, ``u_t=(x1-x0)+sigma_dot*eps`` 를 만든다.

        파라미터:
            batch (AlignmentBatch): 쿼리/레퍼런스/포켓 정보를 담은 입력 배치.
            target_query_pos (Tensor): 형상 ``[N,3]``의 타겟 쿼리 좌표(x1).
            t_graph (Tensor): 형상 ``[num_graphs]``의 그래프별 시간 값.

        반환:
            tuple[Tensor, Tensor]: ``(x_t, u_t)``. 각각 형상 ``[N,3]``의 중간 포즈와 목표 속도.

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 학습 상태 생성의 공개 진입점.
        """
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
        """예측 속도와 목표 속도의 그래프 평균 MSE 손실을 계산한다.

        한 줄 요약:
            원자별 속도 제곱오차를 그래프 단위로 평균낸 뒤, 그래프 간 평균을 반환한다.

        생성이유:
            분자마다 원자 수가 다르므로 단순 원자 평균은 큰 분자에 편향된다.
            먼저 그래프별로 평균낸 뒤 그래프 간 평균을 내어 분자 크기에 불변한 손실을 만든다.

        역할:
            플로우 매칭의 회귀 손실(목표 속도장 예측 오차)을 스칼라로 산출한다.

        메커니즘(핵심 연산/알고리즘):
            - 원자별 제곱오차 ``sum((pred_v - target_u)^2, dim=-1)`` 계산.
            - ``index_add_`` 로 그래프별 오차 합과 원자 수를 누적해 그래프 평균을 구한다.
            - 그래프가 없으면 ``0.0`` 을 곱해 그래프와 연결된 0 손실을 반환한다.

        파라미터:
            pred_v (Tensor): 형상 ``[N,3]``의 모델 예측 속도.
            target_u (Tensor): 형상 ``[N,3]``의 목표 속도 ``u_t``.
            batch_index (Tensor): 형상 ``[N]``의 그래프 소속 인덱스.

        반환:
            Tensor: 스칼라 손실(그래프 평균 MSE).

        파이프라인 단계:
            4단계(학습 목표/확률경로) - 플로우 매칭 손실 계산.
        """
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
    """한 스텝 추정 엔드포인트 x1_hat에 대한 결합 길이 정규화.

    한 줄 요약:
        예측 속도로 한 스텝 외삽한 엔드포인트의 결합 길이를 참값에 맞추는 보조 손실.

    생성이유:
        플로우 매칭 속도 손실만으로는 결합 길이 같은 국소 기하구조가 미세하게
        틀어질 수 있다. 추정 엔드포인트의 결합 길이를 정규화해 화학적으로 타당한
        구조를 유도하기 위함이다.

    역할:
        현재 상태 ``x_t`` 와 예측 속도 ``pred_v`` 로 엔드포인트 ``x1_hat`` 를 추정하고,
        결합 길이의 제곱오차를 반환한다.

    메커니즘(핵심 연산/알고리즘):
        - 결합 정보(``query_bond_index``/``query_bond_length``)가 없거나 비면 0 손실 반환.
        - 노드 시간 ``t_node`` 로 ``x1_hat = x_t + (1 - t)*pred_v`` 외삽.
        - 결합 양 끝 원자 거리 ``||x1_hat[i]-x1_hat[j]||`` 와 목표 길이의 MSE 계산.

    파라미터:
        pred_v (Tensor): 형상 ``[N,3]``의 모델 예측 속도.
        x_t (Tensor): 형상 ``[N,3]``의 현재 중간 상태.
        t_graph (Tensor): 형상 ``[num_graphs]``의 그래프별 시간 값.
        batch (AlignmentBatch): 결합 인덱스/목표 길이와 그래프 인덱스를 제공하는 배치.

    반환:
        Tensor: 스칼라 결합 길이 정규화 손실(결합 정보 없으면 0).

    파이프라인 단계:
        4단계(학습 목표/확률경로) - 결합 길이 보조 정규화 손실.
    """
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
    """한 번의 플로우 매칭 학습 스텝 손실을 계산한다.

    한 줄 요약:
        시간 샘플링 -> 학습 상태 생성 -> 모델 속도 예측 -> 손실 계산을 한 번에 수행한다.

    생성이유:
        학습 루프(train_step)에서 호출하는 단일 진입점으로, 플로우 매칭 손실과
        선택적 결합 정규화 손실을 합산해 역전파할 스칼라 손실을 제공하기 위함이다.

    역할:
        배치를 검증하고, 매처로 ``(x_t, u_t)`` 를 만들고, ``x_t`` 를 모델에 통과시켜
        예측 속도를 얻은 뒤 플로우 매칭 손실(필요 시 결합 손실 가산)을 반환한다.

    메커니즘(핵심 연산/알고리즘):
        - 그래프 수 산정 후 ``matcher.sample_time`` 으로 시간 샘플링.
        - ``matcher.build_training_state`` 로 ``x_t, u_t`` 생성.
        - ``x_t`` 를 좌표로 갖는 새 ``AlignmentBatch`` 를 만들어 모델 호출 -> ``pred_v``.
        - ``matcher.loss`` 로 플로우 매칭 손실 계산.
        - ``lambda_bond > 0`` 이면 ``endpoint_bond_length_loss`` 를 가중 합산.

    파라미터:
        model (ETFlowAlignModel): 속도장을 예측하는 학습 모델.
        matcher (AlignmentFlowMatcher): 확률경로/목표 속도 생성기.
        batch (AlignmentBatch): 입력 배치(쿼리/레퍼런스/포켓/결합 정보).
        target_query_pos (Tensor): 형상 ``[N,3]``의 타겟 쿼리 좌표(x1).
        lambda_bond (float): 결합 길이 정규화 손실의 가중치(0 이하이면 미적용).

    반환:
        Tensor: 스칼라 학습 손실(플로우 매칭 손실, 선택적 결합 손실 포함).

    파이프라인 단계:
        4단계(학습 목표) -> 5단계(학습 루프 train_step) 연결 지점.
    """
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