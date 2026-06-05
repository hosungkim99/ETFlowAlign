"""안전한 가이던스 주입 정책을 갖춘 ETFlowAlign용 ODE 샘플러."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel

GuidanceFn = Callable[[AlignmentBatch, Tensor, Tensor], Tensor]


@dataclass
class ODESamplerConfig:
    """ODE 추론을 위한 설정.

    한 줄 요약:
        ``ETFlowAlignSampler``의 ODE 적분 스텝 수·솔버·가이던스 정책을 모은 dataclass.

    생성이유:
        추론 시 적분 해상도(n_steps), 솔버 종류(Euler/Heun), 가이던스 주입 방식과
        안전 클리핑 등 다수의 추론 하이퍼파라미터를 하나로 묶어 재현 가능한
        샘플링 구성을 제공하기 위함이다.

    역할:
        학습된 모델로 x0->x1 포즈를 적분 생성할 때 필요한 모든 설정값을 보관한다.

    정의되는 주요 변수(dataclass 필드):
        n_steps (int): t_start에서 t_end까지의 적분 스텝 수(적분 해상도).
        t_start (float): 적분 초기 시간.
        t_end (float): 적분 최종 시간.
        solver (Literal): 수치 적분 기법. "euler"=1차, "heun"=2차(예측-보정).
        guidance_scale (float): 외부 가이던스에 적용하는 전역 배율(0 이하이면 비활성).
        guidance_mode (Literal): 가이던스 주입 방식.
            "vector_field"=속도에 직접 가산, "predictor_corrector"=상태 보정 스텝으로 적용.
        max_guidance_norm (float): 가이던스 벡터에 적용되는 원자당 노름 상한(폭주 방지).
        pc_corrector_step_scale (float): predictor-corrector 보정의 스텝 배율.
        adaptive_dt (bool): 속도 노름 기반의 단순 적응형 스텝 배율 활성화 여부.
        adaptive_dt_min_scale (float): 적응형 dt의 최소 배율.
        adaptive_dt_max_scale (float): 적응형 dt의 최대 배율.
        log_trajectory (bool): 적분 궤적 로깅 활성화 여부.
        max_trace_steps (int): 궤적 추적 시 기록할 최대 스텝 수.

    파이프라인 단계:
        7단계(추론 ODE 적분) - 샘플러 동작을 제어하는 설정.
    """

    n_steps: int = 50
    t_start: float = 0.0
    t_end: float = 1.0
    solver: Literal["euler", "heun"] = "heun"
    guidance_scale: float = 0.0
    guidance_mode: Literal["vector_field", "predictor_corrector"] = "vector_field"
    max_guidance_norm: float = 5.0
    pc_corrector_step_scale: float = 1.0
    adaptive_dt: bool = False
    adaptive_dt_min_scale: float = 0.1
    adaptive_dt_max_scale: float = 2.0
    log_trajectory: bool = False
    max_trace_steps: int = 2000


class ETFlowAlignSampler:
    """정렬 플로우 매칭을 위한 ODE 샘플러.

    가이던스 모드:
        - vector_field: 상태 업데이트 전에 가이던스를 속도에 직접 더한다.
        - predictor_corrector: 가이던스를 별도의 보정 스텝으로 적용한다.

    한 줄 요약:
        학습된 속도장 모델로 확률흐름 ODE를 적분해 소스 포즈 x0에서 타겟 포즈 x1을 생성한다.

    생성이유:
        flow matching으로 학습된 속도장은 그 자체로 포즈를 주지 않는다. 시간 0->1로
        ODE를 적분해야 정렬된 포즈가 나오므로, 안전 가드(비유한 값 검출)와 선택적
        가이던스 주입을 갖춘 전용 샘플러가 필요하다.

    역할:
        - 매 스텝 모델 속도를 평가하고(``_model_v``) 선택적 가이던스를 주입(``_apply_guidance``).
        - Euler/Heun 솔버로 상태를 적분(``sample``)하여 최종 포즈를 반환한다.

    정의되는 주요 변수(속성):
        model (ETFlowAlignModel): 시점별 속도장을 예측하는 학습된 모델.
        config (ODESamplerConfig): 스텝 수·솔버·가이던스 정책 등 추론 설정.

    파이프라인 단계:
        7단계(추론 ODE 적분) - 학습 모델로 포즈를 생성한다.
    """

    def __init__(self, model: ETFlowAlignModel, config: ODESamplerConfig) -> None:
        """모델과 설정을 보관해 샘플러를 초기화한다.

        한 줄 요약:
            속도장 모델과 ODE 설정을 받아 샘플러 인스턴스를 구성한다.

        생성이유:
            추론 동안 반복 호출할 모델과 설정을 한 객체에 묶어 상태를 단순화하기 위함이다.

        역할:
            전달받은 모델과 설정을 인스턴스 속성으로 저장한다.

        메커니즘(핵심 연산/알고리즘):
            ``self.model``, ``self.config`` 에 인자를 그대로 대입한다.

        파라미터:
            model (ETFlowAlignModel): 시점별 속도장을 예측하는 학습된 모델.
            config (ODESamplerConfig): ODE 적분/가이던스 설정.

        생성 속성:
            self.model (ETFlowAlignModel): 속도 평가에 사용하는 모델.
            self.config (ODESamplerConfig): 적분/가이던스 동작을 제어하는 설정.

        파이프라인 단계:
            7단계(추론 ODE 적분) - 샘플러 초기화.
        """
        self.model = model
        self.config = config

    def _clip_guidance(self, g: Tensor) -> Tensor:
        """가이던스 벡터의 원자당 노름을 상한으로 제한한다.

        한 줄 요약:
            각 원자 가이던스 벡터의 크기를 ``max_guidance_norm`` 이하로 클리핑한다.

        생성이유:
            가이던스가 과도하게 크면 ODE 적분이 불안정해지고 좌표가 폭주(비유한 값)할 수
            있으므로, 방향은 유지한 채 크기만 제한해 안정성을 확보한다.

        역할:
            입력 가이던스 텐서를 원자별 노름 상한으로 스케일다운한 텐서를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            원자별 노름을 구하고(0 나눗셈 방지 클램프), ``max_guidance_norm/norm`` 을
            ``max=1.0`` 으로 클램프해 노름이 상한을 넘을 때만 축소되도록 곱한다.

        파라미터:
            g (Tensor): 형상 ``[N,3]``의 원자별 가이던스 벡터.

        반환:
            Tensor: 형상 ``[N,3]``의 노름 제한된 가이던스 벡터.

        파이프라인 단계:
            7단계(추론 ODE 적분) - 가이던스 안전 클리핑.
        """
        norm = torch.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)
        return g * (self.config.max_guidance_norm / norm).clamp(max=1.0)

    def _model_v(self, batch: AlignmentBatch, x: Tensor, t_graph: Tensor) -> Tensor:
        """현재 좌표 ``x`` 와 시간에서 모델 속도장을 평가한다.

        한 줄 요약:
            좌표를 ``x`` 로 교체한 배치를 모델에 통과시켜 시점 속도장을 얻는다.

        생성이유:
            ODE 적분 매 스텝마다 변하는 것은 쿼리 좌표뿐이므로, 입력 배치의 조건
            필드(레퍼런스/포켓/속성)는 유지한 채 좌표만 갱신해 속도를 평가해야 한다.

        역할:
            ``x`` 를 쿼리 좌표로 갖는 새 ``AlignmentBatch`` 를 만들어 모델 속도를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            원본 배치의 모든 조건 필드를 복사하되 ``query_pos`` 만 ``x`` 로 바꾼
            배치를 구성하고 ``self.model(cur, t_graph=t_graph)`` 를 호출한다.

        파라미터:
            batch (AlignmentBatch): 조건 필드를 제공하는 원본 배치.
            x (Tensor): 형상 ``[N,3]``의 현재 쿼리 좌표.
            t_graph (Tensor): 형상 ``[num_graphs]``의 현재 시점.

        반환:
            Tensor: 형상 ``[N,3]``의 모델 예측 속도장.

        파이프라인 단계:
            7단계(추론 ODE 적분) - 스텝별 속도장 평가.
        """
        cur = AlignmentBatch(
            query_pos=x,
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
        )
        return self.model(cur, t_graph=t_graph)

    def _apply_guidance(
        self,
        batch: AlignmentBatch,
        x: Tensor,
        t_graph: Tensor,
        v: Tensor,
        guidance_fn: Optional[GuidanceFn],
        dt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """선택적 외부 가이던스를 속도 또는 상태에 주입한다.

        한 줄 요약:
            가이던스 함수를 평가·클리핑한 뒤 모드에 따라 속도나 좌표를 보정한다.

        생성이유:
            추론 시 도킹 스코어·포켓 제약 등 외부 신호로 샘플 궤적을 유도하려면,
            학습된 속도장에 안전하게 외부 가이던스를 결합하는 통합 지점이 필요하다.

        역할:
            가이던스가 비활성이면 입력을 그대로 반환하고, 활성이면 모드별로
            보정된 ``(x, v)`` 를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            - ``guidance_fn`` 이 없거나 scale<=0이면 ``(x, v)`` 그대로 반환.
            - 현재 좌표 ``x`` 로 배치를 구성해 ``guidance_fn`` 평가 후 ``_clip_guidance`` 클리핑.
            - "vector_field": ``v + guidance_scale*g`` 로 속도 보정(좌표는 유지).
            - "predictor_corrector": ``dt`` 필수, ``x + pc_scale*scale*dt*g`` 로 좌표 보정(속도 유지).

        파라미터:
            batch (AlignmentBatch): 조건 필드를 제공하는 원본 배치.
            x (Tensor): 형상 ``[N,3]``의 현재 좌표.
            t_graph (Tensor): 형상 ``[num_graphs]``의 현재 시점.
            v (Tensor): 형상 ``[N,3]``의 현재 속도장.
            guidance_fn (Optional[GuidanceFn]): 가이던스 벡터를 산출하는 콜러블(없으면 무동작).
            dt (Tensor | None): predictor_corrector 모드의 스텝 크기. 해당 모드에서 필수.

        반환:
            tuple[Tensor, Tensor]: 모드에 따라 보정된 ``(x, v)``.

        파이프라인 단계:
            7단계(추론 ODE 적분) - 선택적 가이던스 주입.
        """
        if guidance_fn is None or self.config.guidance_scale <= 0.0:
            return x, v

        cur = AlignmentBatch(
            query_pos=x,
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
        )
        g = self._clip_guidance(guidance_fn(cur, t_graph, v))

        if self.config.guidance_mode == "vector_field":
            return x, v + self.config.guidance_scale * g

        # predictor-corrector: 실제 dt로 스케일된 x 상태의 명시적 보정.
        if dt is None:
            raise ValueError("predictor_corrector guidance requires dt.")
        x_corrected = x + self.config.pc_corrector_step_scale * self.config.guidance_scale * dt * g
        return x_corrected, v

    @torch.no_grad()
    def sample(self, batch: AlignmentBatch, x0: Tensor, guidance_fn: Optional[GuidanceFn] = None) -> Tensor:
        """확률흐름 ODE를 적분해 소스 x0에서 최종 정렬 포즈를 생성한다.

        한 줄 요약:
            시간 격자를 따라 Euler/Heun 솔버로 속도장을 적분하여 x1 포즈를 만든다.

        생성이유:
            학습된 속도장으로부터 실제 정렬 포즈를 얻는 추론의 핵심 절차로,
            안전 가드와 선택적 가이던스를 포함한 단일 공개 진입점이 필요하다.

        역할:
            x0를 초기 상태로 ``n_steps`` 만큼 적분 루프를 돌며 가이던스를 주입하고,
            비유한 값 발생 시 예외를 던지며, 최종 좌표를 반환한다.

        메커니즘(핵심 연산/알고리즘):
            - ``[t_start, t_end]`` 를 ``n_steps+1`` 격자로 나눈다.
            - 각 스텝: 모델 속도 ``v`` 평가 -> 유한성 검사 -> 가이던스 주입.
            - Euler: ``x += dt*v``.
            - Heun: predictor ``x_pred = x + dt*v`` 후(필요 시 PC 보정), 다음 시점 속도
              ``v_next`` 로 ``x += 0.5*dt*(v + v_next)`` (사다리꼴 보정).
            - 매 스텝 좌표 유한성 검사로 폭주를 조기 차단한다.

        파라미터:
            batch (AlignmentBatch): 조건 필드(레퍼런스/포켓/속성)를 제공하는 배치.
            x0 (Tensor): 형상 ``[N,3]``의 초기 소스 포즈.
            guidance_fn (Optional[GuidanceFn]): 선택적 외부 가이던스 콜러블.

        반환:
            Tensor: 형상 ``[N,3]``의 적분 완료된 최종 포즈(x1 추정).

        예외:
            FloatingPointError: 속도 또는 좌표가 비유한 값이 될 때 발생.

        파이프라인 단계:
            7단계(추론 ODE 적분) - 포즈 생성의 공개 진입점.
        """
        num_graphs = int(batch.query_batch.max().item()) + 1
        t_grid = torch.linspace(self.config.t_start, self.config.t_end, self.config.n_steps + 1, device=x0.device)

        x = x0.clone()
        for i in range(self.config.n_steps):
            t = t_grid[i]
            dt = t_grid[i + 1] - t
            t_graph = torch.full((num_graphs,), float(t), device=x.device)

            v = self._model_v(batch, x, t_graph)
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"Non-finite velocity at step={i}, t={float(t):.6f}.")
            x, v = self._apply_guidance(batch, x, t_graph, v, guidance_fn, dt=dt)

            if self.config.solver == "euler":
                x = x + dt * v
            else:
                x_pred = x + dt * v
                t_next = torch.full((num_graphs,), float(t_grid[i + 1]), device=x.device)

                if self.config.guidance_mode == "predictor_corrector" and guidance_fn is not None and self.config.guidance_scale > 0.0:
                    # (a) predictor_corrector 모드: 보정된 predictor 상태를 명시적으로 적용한다.
                    v_pred = self._model_v(batch, x_pred, t_next)
                    x_pred, _ = self._apply_guidance(batch, x_pred, t_next, v_pred, guidance_fn, dt=dt)

                v_next = self._model_v(batch, x_pred, t_next)
                if not torch.isfinite(v_next).all():
                    raise FloatingPointError(
                    f"Non-finite coordinates during ODE integration at step={i}, t={float(t):.6f}. "
                    "Check velocity scale, solver step size, batch fields, or model rollout stability."
                )
                _, v_next = self._apply_guidance(batch, x_pred, t_next, v_next, guidance_fn, dt=dt)
                x = x + 0.5 * dt * (v + v_next)

            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite coordinates during ODE integration. Reduce guidance scale.")

        return x