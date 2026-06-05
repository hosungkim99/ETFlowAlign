# etflowalign/scripts/diagnose_vector_field.py
"""ETFlowAlign 체크포인트에 대한 간단한 벡터 필드 진단 스크립트.

Example:
    python -m etflowalign.scripts.diagnose_vector_field \
      --checkpoint /path/to/checkpoint_best.pt \
      --train-data /path/to/diffalign_example_train.pt \
      --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from etflowalign.data import load_alignment_batch_from_pt
from etflowalign.model import AlignmentBatch, ETFlowAlignModel
from etflowalign.flow_matching import AlignmentFlowMatcher, FlowMatchingConfig


def build_argparser() -> argparse.ArgumentParser:
    """벡터 필드 진단 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        벡터 필드(속도장) 진단에 필요한 명령행 인자를 정의한 ArgumentParser를 만든다.
    생성이유:
        체크포인트/학습 배치/디바이스/평가할 flow time 목록/CSV 출력 경로를 CLI로
        지정하기 위한 인자 사양을 한 곳에 모으기 위함.
    역할:
        --checkpoint, --train-data, --device, --t-values, --csv-out 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        7단계(추론) 진단 - 속도장 점검 도구의 인자 정의.
    """
    parser = argparse.ArgumentParser(description="Diagnose ETFlowAlign vector field.")

    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path.")
    parser.add_argument("--train-data", required=True, help="Training batch .pt path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Flow times to evaluate.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional CSV output path.",
    )

    return parser


def make_step_batch(batch: AlignmentBatch, x_t: torch.Tensor) -> AlignmentBatch:
    """주어진 보간 좌표 x_t를 query_pos로 갖는 새 AlignmentBatch를 만든다.

    한 줄 요약:
        조건 텐서는 유지하고 query_pos만 x_t로 교체한 AlignmentBatch를 생성한다.
    생성이유:
        특정 flow time t에서 모델 속도장을 평가하려면 보간 상태 x_t를 입력 좌표로
        하는 배치가 필요하지만, 레퍼런스/포켓/노드 속성 등 조건은 그대로 두어야 한다.
    역할:
        원본 배치의 모든 조건 필드를 복사하고 query_pos만 x_t로 설정한다.
    메커니즘:
        AlignmentBatch를 새로 생성하며 query_pos에 x_t를, 나머지 필드에는 기존 batch 값을 전달한다.
    파라미터:
        batch (AlignmentBatch): 조건 정보를 담은 원본 배치.
        x_t (torch.Tensor): flow time t에서의 보간 쿼리 좌표 [N,3].
    반환:
        AlignmentBatch: query_pos만 교체된 새 배치.
    파이프라인 단계:
        7단계(추론) 진단 - 시점별 모델 호출용 배치 구성.
    """
    return AlignmentBatch(
        query_pos=x_t,
        query_atom_type=batch.query_atom_type,
        query_batch=batch.query_batch,
        reference_pos=batch.reference_pos,
        reference_atom_type=batch.reference_atom_type,
        reference_batch=batch.reference_batch,
        pocket_pos=batch.pocket_pos,
        pocket_batch=batch.pocket_batch,
        query_node_attr=batch.query_node_attr,
        reference_node_attr=batch.reference_node_attr,
    )


def main() -> None:
    """CLI 진입점: 여러 flow time에서 모델 속도장과 정답 속도장의 오차를 측정한다.

    한 줄 요약:
        체크포인트를 불러와 지정한 t 값들에서 모델 예측 속도와 정답 속도의 오차 지표를 계산/출력/저장한다.
    생성이유:
        학습된 모델이 흐름(flow)의 각 시점에서 올바른 속도장을 학습했는지 정량 점검하여,
        샘플링 실패의 원인이 속도장 자체에 있는지 진단하기 위함.
    역할:
        체크포인트와 학습 배치(타겟 포함)를 로드해 모델과 FlowMatcher를 복원하고, 각 t 값마다
        보간 상태 x_t와 정답 속도 true_u를 만들어 모델 예측 pred_u와 비교한 MSE/RMSE/평균·최대 오차를
        출력하고 옵션 시 CSV로 저장한다.
    메커니즘:
        torch.load로 ckpt 로드, load_alignment_batch_from_pt로 배치/타겟/메타데이터 로드(require_target=True),
        ETFlowAlignModel 복원 후 eval. flow_args의 fixed_t를 None으로 만들어 AlignmentFlowMatcher 생성.
        각 t에 대해 t_graph를 채우고 matcher.build_training_state로 (x_t, true_u)를 얻은 뒤,
        make_step_batch로 모델을 호출(no_grad)해 pred_u를 구하고 오차 통계를 계산한다.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 표준출력과(지정 시) CSV로 출력된다.
    파이프라인 단계:
        7단계(추론) 진단 - 속도장 정확도 점검(학습목표/모델 sanity check 겸용).
    """
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print("checkpoint:", args.checkpoint)
    print("train_data:", args.train_data)
    print("device:", device)
    print("model_args:", ckpt["model_args"])
    print("flow_args:", ckpt["flow_args"])
    print("best_loss:", ckpt.get("best_loss"))
    print("best_step:", ckpt.get("best_step"))
    print("final_loss:", ckpt.get("final_loss"))

    batch, target_query_pos, metadata = load_alignment_batch_from_pt(
        args.train_data,
        require_target=True,
        device=device,
    )
    assert target_query_pos is not None

    print("metadata source:", metadata.get("source") if isinstance(metadata, dict) else None)
    print("query_pos:", batch.query_pos.shape, batch.query_pos.dtype)
    print(
        "query_node_attr:",
        None if batch.query_node_attr is None else (batch.query_node_attr.shape, batch.query_node_attr.dtype),
    )
    print("reference_pos:", None if batch.reference_pos is None else batch.reference_pos.shape)
    print(
        "reference_node_attr:",
        None if batch.reference_node_attr is None else (batch.reference_node_attr.shape, batch.reference_node_attr.dtype),
    )
    print("pocket_pos:", None if batch.pocket_pos is None else batch.pocket_pos.shape)
    print("target_query_pos:", target_query_pos.shape, target_query_pos.dtype)

    model = ETFlowAlignModel(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    flow_args = ckpt["flow_args"].copy()
    flow_args["fixed_t"] = None
    matcher = AlignmentFlowMatcher(FlowMatchingConfig(**flow_args))

    rows = []

    num_graphs = int(batch.query_batch.max().item()) + 1
    print("source max abs diff from batch.query_pos:", (batch.query_pos - batch.query_pos).abs().max().item())

    for tval in args.t_values:
        t_graph = torch.full(
            (num_graphs,),
            float(tval),
            device=device,
            dtype=batch.query_pos.dtype,
        )

        x_t, true_u = matcher.build_training_state(
            batch=batch,
            target_query_pos=target_query_pos,
            t_graph=t_graph,
        )

        step_batch = make_step_batch(batch, x_t)

        with torch.no_grad():
            pred_u = model(step_batch, t_graph)

        err = pred_u - true_u
        mse = err.pow(2).sum(dim=-1).mean().item()
        rmse = mse ** 0.5
        max_err = err.norm(dim=-1).max().item()
        mean_err = err.norm(dim=-1).mean().item()

        print(f"t={tval:.2f}")
        print(f"  vector MSE loss-like: {mse:.6f}")
        print(f"  vector RMSE: {rmse:.6f} Å")
        print(f"  mean atom vector error: {mean_err:.6f} Å")
        print(f"  max atom vector error: {max_err:.6f} Å")

        rows.append(
            {
                "t": float(tval),
                "vector_mse": mse,
                "vector_rmse_A": rmse,
                "mean_atom_vector_error_A": mean_err,
                "max_atom_vector_error_A": max_err,
            }
        )

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("csv saved to:", out_path)


if __name__ == "__main__":
    main()
    
'''
TRAIN_ROOT=/home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests
INPUT_PT=$TRAIN_ROOT/00_inputs/diffalign_example/diffalign_example_train.pt
OUT_ROOT=$TRAIN_ROOT/04_equivariant_basis_head/nodeattr

python -m etflowalign.scripts.diagnose_vector_field \
  --checkpoint $OUT_ROOT/checkpoints/etflowalign_basis_nodeattr_randomt_3k_best.pt \
  --train-data $INPUT_PT \
  --device cuda \
  --t-values 0.0 0.25 0.5 0.75 1.0 \
  --csv-out $OUT_ROOT/diagnostics/diagnose_basis_nodeattr_randomt_3k_best.csv
'''
