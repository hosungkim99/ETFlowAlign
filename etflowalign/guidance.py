"""배치 안전 인터페이스 검사를 갖춘 포켓 인식 가이던스 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch

UFFEnergyFn = Callable[[Tensor, AlignmentBatch, "UFFGuidanceConfig"], Tensor]


@dataclass
class UFFGuidanceConfig:
    """UFF/포켓 기반 가이던스의 하이퍼파라미터를 담는 설정 컨테이너.

    생성이유:
        가이던스 강도 등 조정 가능한 값을 코드에 흩어두지 않고 한 곳에 모아
        샘플링 호출 간 일관되게 전달하기 위해 dataclass로 분리한다.

    역할:
        UFFPocketGuidance가 에너지/그래디언트를 계산할 때 참조하는 설정값을
        보관한다.

    필드:
        scale (float): 가이던스 벡터의 스케일 계수. 음의 에너지 그래디언트에
            곱해져 가이던스 세기를 조절한다. 기본값 1.0.

    파이프라인 단계:
        9단계(가이던스) - 설정 보조.
    """

    scale: float = 1.0


class UFFPocketGuidance:
    """그래프 간 비물리적 결합을 방지하는 간단한 미분 가능 포켓 인식 가이던스.

    생성이유:
        flow matching 샘플링만으로는 원자 간 충돌 등 물리적으로 부자연스러운
        포즈가 나올 수 있다. 에너지의 음의 그래디언트를 가이던스 벡터로
        주입하여 더 물리적으로 타당한 방향으로 적분을 유도하기 위해 만든다.

    역할:
        가이던스(9단계)의 핵심 클래스. 좌표로부터 에너지를 계산하고, 그
        그래디언트를 이용해 샘플링 중 주입할 ``[Nq, 3]`` 가이던스 텐서를
        생성하는 호출 가능한(callable) 객체로 동작한다.

    정의되는 주요 변수와 의미:
        config (UFFGuidanceConfig): 가이던스 스케일 등 하이퍼파라미터.
        _backend_energy_fn (Optional[UFFEnergyFn]): 외부에서 주입 가능한
            대체 에너지 함수. 주어지면 기본 내장 에너지 대신 사용된다.

    파이프라인 단계:
        9단계(가이던스).
    """

    def __init__(self, config: Optional[UFFGuidanceConfig] = None, backend_energy_fn: Optional[UFFEnergyFn] = None) -> None:
        """가이던스 객체의 설정과 백엔드 에너지 함수를 초기화한다.

        생성 속성:
            self.config (UFFGuidanceConfig): 전달된 config가 없으면 기본
                UFFGuidanceConfig를 생성해 사용한다.
            self._backend_energy_fn (Optional[UFFEnergyFn]): 외부 에너지 함수.
                None이면 내장 거리 기반 에너지(energy)를 사용한다.

        파라미터:
            config (Optional[UFFGuidanceConfig]): 가이던스 하이퍼파라미터.
                None이면 기본값으로 생성된다.
            backend_energy_fn (Optional[UFFEnergyFn]): (x, batch, config)를 받아
                스칼라 텐서 에너지를 반환하는 대체 에너지 콜백.

        파이프라인 단계:
            9단계(가이던스) - 객체 초기화.
        """
        self.config = config or UFFGuidanceConfig()
        self._backend_energy_fn = backend_energy_fn

    def energy(self, x: Tensor, batch: AlignmentBatch) -> Tensor:
        """좌표로부터 그래프별로 가산되는 미분 가능한 스칼라 에너지를 계산한다.

        생성이유:
            가이던스 벡터는 에너지의 그래디언트이므로, 먼저 미분 가능한 에너지
            함수가 필요하다. 그래프(분자)별로 분리해 계산함으로써 서로 다른
            그래프의 원자들이 인접 거리로 결합되는 것을 막는다.

        역할:
            기본 폴백 에너지를 정의한다. 외부 백엔드 에너지 함수가 주입된
            경우 그 결과를 그대로 사용하고, 아니면 그래프 내부 원자 쌍의 역거리
            기반 척력 에너지를 합산한다.

        메커니즘:
            _backend_energy_fn이 있으면 호출하고 반환값이 텐서인지 검사한 뒤
            반환한다. 없으면 query_batch의 각 그래프 id에 대해 해당 원자 좌표
            xg를 모으고, 원자가 2개 이상일 때만 쌍거리(cdist)를 계산한다.
            자기 자신과의 거리(대각)를 1e6으로 부풀려 제외하고,
            1/clamp(d,1e-3)의 평균을 누적해 총 에너지로 더한다.

        파라미터:
            x (Tensor): 현재 query 좌표 ``[Nq, 3]``. 그래디언트 추적 대상.
            batch (AlignmentBatch): 그래프 분할 정보(query_batch)를 제공하는 배치.

        반환:
            Tensor: 그래프별 기여가 합산된 스칼라 에너지 텐서.

        파이프라인 단계:
            9단계(가이던스) - 에너지 계산.
        """
        if self._backend_energy_fn is not None:
            out = self._backend_energy_fn(x, batch, self.config)
            if not torch.is_tensor(out):
                raise TypeError("Backend energy function must return a torch.Tensor scalar.")
            return out

        total_energy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for graph_id in batch.query_batch.unique(sorted=True):
            graph_mask = batch.query_batch == graph_id
            xg = x[graph_mask]
            if xg.size(0) <= 1:
                continue
            d = torch.cdist(xg, xg)
            d = d + torch.eye(xg.size(0), device=x.device, dtype=x.dtype) * 1e6
            total_energy = total_energy + (1.0 / d.clamp_min(1e-3)).mean()
        return total_energy

    def __call__(self, batch: AlignmentBatch, t_graph: Tensor, v: Tensor) -> Tensor:
        """샘플러가 주입할 정확히 ``[Nq, 3]`` 형상의 가이던스 텐서를 계산해 반환한다.

        생성이유:
            샘플러는 매 적분 스텝에서 GuidanceFn 시그니처(batch, t_graph, v)로
            가이던스를 호출한다. 이 객체를 그 인터페이스에 맞춰 호출 가능하게
            만들어 샘플링 루프에 그대로 끼워 넣기 위함이다.

        역할:
            현재 배치 좌표에 대한 에너지 그래디언트를 구하고, 음의 그래디언트에
            스케일을 곱해 가이던스 벡터를 만든다. 출력 형상이 벡터장 v와
            일치하는지 검증한다.

        메커니즘:
            t_graph는 사용하지 않으므로 del로 명시 제거한다. batch.query_pos를
            detach·clone 후 requires_grad_(True)로 만들어 x를 구성하고,
            energy(x, batch)에 대해 torch.autograd.grad로 그래디언트를 계산한다
            (그래프 보존 없음). guidance = -scale * grad. guidance의 형상이 v와
            다르면 ValueError로 실패시키고, 같으면 guidance를 반환한다.

        파라미터:
            batch (AlignmentBatch): 현재 좌표(query_pos)와 그래프 정보를 가진 배치.
            t_graph (Tensor): 그래프별 적분 시각. 본 구현에서는 사용하지 않음.
            v (Tensor): 현재 벡터장 ``[Nq, 3]``. 반환 형상 검증의 기준이 된다.

        반환:
            Tensor: 가이던스 벡터 ``[Nq, 3]`` (= -scale * 에너지 그래디언트).

        파이프라인 단계:
            9단계(가이던스) - 샘플링 중 가이던스 벡터 주입.
        """
        del t_graph
        x = batch.query_pos.detach().clone().requires_grad_(True)
        grad = torch.autograd.grad(self.energy(x=x, batch=batch), x, create_graph=False, retain_graph=False)[0]
        guidance = -self.config.scale * grad

        if guidance.shape != v.shape:
            raise ValueError(
                f"GuidanceFn must return Tensor[Nq,3]; expected {tuple(v.shape)} got {tuple(guidance.shape)}."
            )
        return guidance