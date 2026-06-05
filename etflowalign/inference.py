"""ETFlowAlign 추론 스크립트 및 API."""

from __future__ import annotations

import argparse
from typing import Any,Callable, Optional

import torch
from torch import Tensor

from .data import load_alignment_batch_from_pt
from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig
from .model import AlignmentBatch, ETFlowAlignModel
from .sampler import ETFlowAlignSampler, GuidanceFn, ODESamplerConfig
from .train import make_synthetic_alignment_batch
from .validation import validate_alignment_batch

RankFn = Callable[[Tensor, AlignmentBatch], Tensor]


def generate_candidates(
    sampler: ETFlowAlignSampler,
    batch: AlignmentBatch,
    source_sampler: Callable[[AlignmentBatch], Tensor],
    guidance_fn: Optional[GuidanceFn] = None,
    num_samples: int = 8,
) -> Tensor:
    """엄격한 인터페이스 검사를 통해 정렬된 후보 포즈를 다중 생성한다.

    생성이유:
        추론 시 단일 결정론적 출력만으로는 정렬 품질을 보장하기 어렵다. 서로
        다른 소스(초기 노이즈) 샘플로부터 여러 후보를 만들어 두면, 이후
        rank_candidates 단계에서 가장 좋은 포즈를 골라낼 수 있다.

    역할:
        추론 오케스트레이션(8단계)에서 "후보 생성" 책임을 담당한다.
        ODE 적분 기반 샘플러를 num_samples 회 반복 호출하여 후보 집합을 만든다.

    메커니즘:
        1) validate_alignment_batch로 입력 배치의 인터페이스 정합성을 먼저 검사한다.
        2) num_samples 만큼 반복하면서 source_sampler로 초기 좌표 x0를 뽑는다.
        3) x0의 형상이 batch.query_pos와 정확히 일치하는지 확인하고, 다르면
           ValueError로 즉시 실패시켜 후속 단계의 형상 오류를 차단한다.
        4) sampler.sample로 x0를 query_pos 조건에 맞게 적분하여 후보 포즈를 얻는다.
        5) 모든 후보를 dim=0 축으로 stack 하여 단일 텐서로 반환한다.

    파라미터:
        sampler (ETFlowAlignSampler): ODE 적분으로 포즈를 생성하는 샘플러(7단계).
        batch (AlignmentBatch): query/reference/pocket 좌표 및 조건이 담긴 배치.
        source_sampler (Callable[[AlignmentBatch], Tensor]): 배치를 받아
            query_pos 형상의 초기 좌표 x0를 반환하는 소스 분포 샘플러.
        guidance_fn (Optional[GuidanceFn]): 샘플링 중 주입할 물리/포켓 가이던스
            함수. None이면 가이던스를 적용하지 않는다(9단계).
        num_samples (int): 생성할 후보 포즈의 개수. 기본값 8.

    반환:
        Tensor: 형상 ``[num_samples, Nq, 3]``의 후보 포즈 텐서.

    파이프라인 단계:
        8단계(추론 오케스트레이션) - 후보 생성 담당.
    """
    validate_alignment_batch(batch)
    outputs: list[Tensor] = []
    for _ in range(num_samples):
        x0 = source_sampler(batch)
        if x0.shape != batch.query_pos.shape:
            raise ValueError(
                "source_sampler must return query_pos-matched shape "
                f"{tuple(batch.query_pos.shape)}, got {tuple(x0.shape)}."
            )
        outputs.append(sampler.sample(batch=batch, x0=x0, guidance_fn=guidance_fn))
    return torch.stack(outputs, dim=0)


def rank_candidates(
    candidates: Tensor,
    batch: AlignmentBatch,
    rank_fn: Optional[RankFn] = None,
) -> tuple[Tensor, Tensor]:
    """v0.1 호출당 단일 복합체 유효성 검사를 통해 후보 포즈를 점수 순으로 랭킹한다.

    생성이유:
        generate_candidates로 만든 다수의 후보 중에서 정렬 품질이 가장 좋은
        포즈를 선택해야 한다. 이를 위해 후보별 점수를 매기고 내림차순으로
        정렬하는 통일된 진입점이 필요하다.

    역할:
        추론 오케스트레이션(8단계)에서 "후보 랭킹" 책임을 담당한다. 점수
        함수(rank_fn)를 위임받아 후보를 점수화하고, 점수 높은 순으로 재배열한다.

    메커니즘:
        1) batch.query_batch로부터 그래프(복합체) 개수를 계산한다.
        2) v0.1은 호출당 정확히 1개의 복합체만 지원하므로, 아니면 ValueError.
        3) rank_fn이 주어지면 후보를 점수화하고, 없으면 모든 점수를 0으로 둔다.
        4) 점수 텐서의 형상이 1차원이고 후보 수와 일치하는지 검증한다.
        5) torch.argsort(descending=True)로 내림차순 정렬 인덱스를 얻어
           후보와 점수를 동일 순서로 재배열하여 반환한다.

    파라미터:
        candidates (Tensor): 형상 ``[num_samples, Nq, 3]``의 후보 포즈 텐서.
        batch (AlignmentBatch): 점수 계산 및 복합체 개수 검증에 쓰이는 배치.
        rank_fn (Optional[RankFn]): 후보 전체와 배치를 받아 후보당 점수 하나
            (``Tensor[num_samples]``)를 반환하는 점수 함수. None이면 0점 처리.

    반환:
        tuple[Tensor, Tensor]: (정렬된 후보 ``[num_samples, Nq, 3]``,
            정렬된 점수 ``[num_samples]``). 둘 다 점수 내림차순.

    파이프라인 단계:
        8단계(추론 오케스트레이션) - 후보 랭킹 담당.
    """
    num_graphs = int(batch.query_batch.max().item()) + 1 if batch.query_batch.numel() else 0
    if num_graphs != 1:
        raise ValueError("v0.1 ranking supports exactly one complex per inference call.")

    scores = rank_fn(candidates, batch) if rank_fn is not None else torch.zeros(candidates.size(0), device=candidates.device)
    if scores.dim() != 1 or scores.size(0) != candidates.size(0):
        raise ValueError("rank_fn must return Tensor[num_samples].")

    order = torch.argsort(scores, descending=True)
    return candidates[order], scores[order]


def run_inference(args: argparse.Namespace) -> None:
    """체크포인트 로드부터 샘플링, 랭킹, 저장까지 전체 추론을 수행하는 진입점.

    생성이유:
        학습된 모델로 실제 정렬 포즈를 얻으려면 체크포인트 복원, 샘플러 구성,
        입력 배치 준비, 후보 생성/랭킹, 결과 저장을 하나의 흐름으로 묶어야
        한다. 이 함수가 그 오케스트레이션을 담당한다.

    역할:
        추론 오케스트레이션(8단계)의 최상위 실행 함수. CLI 인자(args)를 받아
        모델·flow matcher·샘플러를 재구성하고, 입력 배치를 로드한 뒤
        generate_candidates와 rank_candidates를 호출하고 결과를 저장/출력한다.

    메커니즘:
        1) args.device로 디바이스를 정하고 체크포인트를 torch.load 한다.
        2) ckpt의 model_args/model_state로 ETFlowAlignModel을 복원하고 eval 모드로.
        3) ckpt의 flow_args로 AlignmentFlowMatcher와 ODESamplerConfig를 구성하여
           ETFlowAlignSampler를 만든다(가이던스 관련 인자 포함).
        4) synthetic_smoke면 합성 배치를 생성하고, 아니면 .pt 파일에서 배치를
           로드한다(추론이므로 require_target=False).
        5) generate_candidates로 후보를 만들고(소스는 flow_matcher.sample_source,
           guidance_fn=None) rank_candidates로 랭킹한다.
        6) 후보 형상과 top 점수를 출력한다.
        7) save_path가 있으면 후보/점수/실행 메타데이터를 torch.save로 저장한다.

    파라미터:
        args (argparse.Namespace): build_argparser로 파싱된 CLI 인자. checkpoint,
            device, num_samples, n_steps, solver, guidance 관련 옵션, 입력 소스
            (synthetic_smoke 또는 input_batch), save_path 등을 포함한다.

    반환:
        None. 결과는 stdout 출력 및 (지정 시) save_path 파일로 부수효과로 남는다.

    파이프라인 단계:
        8단계(추론 오케스트레이션) - 체크포인트 로드→샘플→랭킹→저장 전체 실행.
    """
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = ETFlowAlignModel(**ckpt.get("model_args", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    flow_args = ckpt.get("flow_args", {})
    flow_matcher = AlignmentFlowMatcher(FlowMatchingConfig(**flow_args))
    sampler = ETFlowAlignSampler(
        model=model,
        config=ODESamplerConfig(
            n_steps=args.n_steps,
            solver=args.solver,
            guidance_scale=args.guidance_scale,
            guidance_mode=args.guidance_mode,
            max_guidance_norm=args.max_guidance_norm,
        ),
    )

    input_metadata: dict[str, Any] = {}
    if args.synthetic_smoke:
        batch, _ = make_synthetic_alignment_batch(batch_size=1, n_atoms=args.n_atoms, device=device)
    else:
        batch, _, input_metadata = load_alignment_batch_from_pt(
            args.input_batch,
            require_target=False,
            device=device,
        )

    candidates = generate_candidates(
        sampler=sampler,
        batch=batch,
        source_sampler=lambda b: flow_matcher.sample_source(b),
        guidance_fn=None,
        num_samples=args.num_samples,
    )
    ranked, scores = rank_candidates(candidates, batch)

    print(f"[inference] candidates={ranked.shape}")
    print(f"[inference] top_score={scores[0].item():.6f}")

    if args.save_path:
        run_metadata = {
            "checkpoint": args.checkpoint,
            "input_batch": args.input_batch if args.input_batch else None,
            "synthetic_smoke": bool(args.synthetic_smoke),
            "num_samples": int(args.num_samples),
            "n_steps": int(args.n_steps),
            "solver": args.solver,
            "guidance_scale": float(args.guidance_scale),
            "guidance_mode": args.guidance_mode,
            "source_type": flow_args.get("source_type"),
            "input_metadata": input_metadata,
        }
        torch.save(
            {
                "candidates": ranked.cpu(),
                "scores": scores.cpu(),
                "metadata": run_metadata,
            },
            args.save_path,
        )
        print(f"[inference] saved to: {args.save_path}")


def build_argparser() -> argparse.ArgumentParser:
    """추론 CLI의 인자 파서를 구성하여 반환한다.

    생성이유:
        run_inference가 필요로 하는 모든 옵션(체크포인트 경로, 디바이스, 샘플
        개수, ODE 스텝/솔버, 가이던스 설정, 입력 소스, 저장 경로)을 커맨드라인
        에서 일관되게 받기 위한 단일 정의 지점이 필요하다.

    역할:
        ArgumentParser 객체를 생성하고 각 옵션을 등록한다. main()과
        run_inference가 이 파서로 args를 파싱해 사용한다.

    메커니즘:
        argparse.ArgumentParser를 만들고 add_argument로 checkpoint(필수),
        device, num-samples, n-atoms, n-steps, solver(euler/heun),
        guidance-scale, guidance-mode(vector_field/predictor_corrector),
        max-guidance-norm, synthetic-smoke(플래그), input-batch, save-path를
        등록한다.

    파라미터:
        없음.

    반환:
        argparse.ArgumentParser: 추론 옵션이 모두 등록된 파서 객체.

    파이프라인 단계:
        8단계(추론 오케스트레이션) - CLI 인자 정의 보조.
    """
    p = argparse.ArgumentParser(description="Run ETFlowAlign inference.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--guidance-mode", type=str, default="vector_field", choices=["vector_field", "predictor_corrector"])
    p.add_argument("--max-guidance-norm", type=float, default=5.0)

    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--input-batch", type=str, default="")
    p.add_argument("--save-path", type=str, default="")
    return p


def main() -> None:
    """CLI 실행의 최상위 엔트리포인트.

    생성이유:
        ``python -m etflowalign.inference`` 형태로 모듈을 직접 실행했을 때의
        시작점이 필요하다. 인자 파싱과 입력 소스 상호배타 검증을 한 곳에 모은다.

    역할:
        파서로 인자를 파싱하고, synthetic-smoke와 input-batch 중 정확히 하나만
        지정되었는지 검증한 뒤 run_inference를 호출한다.

    메커니즘:
        build_argparser().parse_args()로 args를 얻는다. synthetic_smoke(bool)와
        input_batch(존재 여부 bool)가 동일하면(둘 다 켜졌거나 둘 다 꺼졌으면)
        ValueError를 발생시켜 입력 소스를 정확히 하나만 강제한다. 이후
        run_inference(args)를 실행한다.

    파라미터:
        없음(커맨드라인 인자에서 직접 읽음).

    반환:
        None.

    파이프라인 단계:
        8단계(추론 오케스트레이션) - CLI 최상위 실행.
    """
    args = build_argparser().parse_args()
    if args.synthetic_smoke == bool(args.input_batch):
        raise ValueError("Provide exactly one of --synthetic-smoke or --input-batch.")
    run_inference(args)


if __name__ == "__main__":
    main()