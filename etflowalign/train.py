"""ETFlowAlign 학습 스크립트 및 유틸리티.

Examples:
    python -m etflowalign.train --synthetic-smoke --steps 2 --batch-size 2 --n-atoms 8 --save-path /tmp/etflowalign_smoke.pt
    python -m etflowalign.train --train-data /path/to/train_batch.pt --steps 1 --save-path /tmp/etflowalign_ckpt.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, optim

from .data import load_alignment_batch_from_pt
from .dataset import AlignmentDataset, sample_training_batch, train_val_split
from .evaluation import evaluate_paths, format_summary
from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class TrainConfig:
    """ETFlowAlign 학습을 위한 최적화 하이퍼파라미터."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_bond: float = 0.0


def build_training_components(
    model: ETFlowAlignModel,
    train_config: TrainConfig,
    fm_config: FlowMatchingConfig,
) -> tuple[AlignmentFlowMatcher, optim.Optimizer]:
    """플로우 매처와 옵티마이저를 생성한다."""
    matcher = AlignmentFlowMatcher(config=fm_config)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    return matcher, optimizer


def train_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    optimizer: optim.Optimizer,
    batch: AlignmentBatch,
    target_query_pos: Tensor,
    lambda_bond: float = 0.0,
) -> float:
    """최적화 스텝을 한 번 실행하고 스칼라 손실값을 반환한다."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = flow_matching_step(
        model,
        matcher,
        batch,
        target_query_pos,
        lambda_bond=lambda_bond,
    )

    loss_check = float(loss.detach())
    if not math.isfinite(loss_check) or abs(loss_check) > 1e4:
        # 비정상(NaN/Inf)이거나 폭발하는 손실값(정상 손실은 ~10-200)을 건너뛴다. 여기서
        # 예외를 발생시키면 backward/step을 우회하므로 하나의 이상 배치가 가중치를 오염시키지
        # 않는다; 호출자는 최적 체크포인트로 롤백하고 학습을 계속한다.
        raise FloatingPointError(f"Non-finite/exploding training loss before backward: {loss_check}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    # 스텝 실행 전에 비정상 그래디언트를 정제한다. clip_grad_norm_은 이를 막지 못한다.
    # NaN/Inf 그래디언트가 하나라도 있으면 total_norm이 NaN이 되어 모든 가중치를 오염시키고,
    # 다음 순전파에서 NaN이 발생해 학습이 종료된다. 잘못된 그래디언트를 0으로 만들면
    # 드물게 발생하는 불안정한 스텝이 크래시 대신 아무 효과 없는 연산으로 처리된다.
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    optimizer.step()

    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            f"Non-finite training loss after optimization step: {loss_value}"
        )

    return loss_value


def make_synthetic_alignment_batch(batch_size: int, n_atoms: int, device: torch.device) -> tuple[AlignmentBatch, Tensor]:
    """스모크 테스트용 간단한 정렬 배치를 생성한다."""
    query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_atoms)
    reference_batch = query_batch.clone()

    reference_pos = 0.7 * torch.randn(batch_size * n_atoms, 3, device=device)
    atom_type = torch.randint(low=0, high=16, size=(batch_size * n_atoms,), device=device)

    translation = 0.25 * torch.randn(batch_size, 3, device=device)
    target_query_pos = reference_pos + translation[query_batch]
    target_query_pos = target_query_pos + 0.05 * torch.sin(target_query_pos * 2.0)

    query_init = reference_pos + 0.8 * torch.randn_like(reference_pos)
    batch = AlignmentBatch(
        query_pos=query_init,
        query_atom_type=atom_type,
        query_batch=query_batch,
        reference_pos=reference_pos,
        reference_atom_type=atom_type,
        reference_batch=reference_batch,
    )
    return batch, target_query_pos


def _clone_state_dict_to_cpu(model: ETFlowAlignModel) -> dict[str, torch.Tensor]:
    """체크포인트 저장을 위해 모델 파라미터를 CPU로 복제한다."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _make_best_checkpoint_path(save_path: str) -> Path:
    """최종 체크포인트 경로로부터 최적 체크포인트 경로를 생성한다."""
    path = Path(save_path)
    if path.suffix == ".pt":
        return path.with_name(f"{path.stem}_best{path.suffix}")
    return Path(f"{save_path}.best.pt")


def run_training(args: argparse.Namespace) -> None:
    """CLI 학습 진입점."""
    device = torch.device(args.device)

    model_args = {
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "use_atom_index_embed": args.use_atom_index_embed,
        "use_direct_vector_head": args.use_direct_vector_head,
        "use_equivariant_basis_head": args.use_equivariant_basis_head,
        "use_rigid_head": args.use_rigid_head,
        "use_pocket_conditioning": args.use_pocket_conditioning,
        "pocket_cutoff": args.pocket_cutoff,
        "pocket_max_neighbors": args.pocket_max_neighbors,
        "use_node_attr": args.use_node_attr,
        "node_attr_dim": args.node_attr_dim,
        "max_atoms": args.max_atoms,
    }

    model = ETFlowAlignModel(**model_args).to(device)

    flow_args = {
        "sigma": args.sigma,
        "source_type": args.source_type,
        "source_noise_scale": args.source_noise_scale,
        "center_source": args.center_source,
        "center_target": args.center_target,
        "use_kabsch_alignment": args.use_kabsch_alignment,
        "fixed_t": args.fixed_t,
        "path_type": args.path_type,
    }

    fm_config = FlowMatchingConfig(**flow_args)

    train_config = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, lambda_bond=args.lambda_bond,)
    matcher, optimizer = build_training_components(model, train_config, fm_config)

    train_args = vars(args).copy()

    dataset: AlignmentDataset | None = None
    sample_generator: torch.Generator | None = None
    val_paths: list[str] = []
    if args.train_dir:
        all_paths = AlignmentDataset.from_directory(args.train_dir, require_target=True).paths
        if args.train_limit > 0:
            all_paths = all_paths[: args.train_limit]
        if args.val_fraction > 0.0:
            train_paths, val_paths = train_val_split(all_paths, args.val_fraction, args.split_seed)
            val_manifest = Path(f"{args.save_path}.val.txt")
            val_manifest.parent.mkdir(parents=True, exist_ok=True)
            val_manifest.write_text("\n".join(val_paths))
            dataset = AlignmentDataset(train_paths, require_target=True)
            print(f"[train] held-out val: {len(val_paths)} complexes (never trained) -> {val_manifest}")
        else:
            dataset = AlignmentDataset(all_paths, require_target=True)
        sample_generator = torch.Generator().manual_seed(0)
        print(
            f"[train] dataset: {len(dataset)} train complexes from {args.train_dir} "
            f"| conditioning={args.conditioning} batch_size={args.batch_size}"
        )
    
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = _make_best_checkpoint_path(args.save_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint(model_state: dict[str, torch.Tensor], extra: dict) -> dict:
        ckpt = {
            "model_state": model_state,
            "model_args": model_args,
            "flow_args": flow_args,
            "train_args": train_args,
        }
        ckpt.update(extra)
        return ckpt
    
    best_loss = float("inf")
    best_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    final_loss: float | None = None
    consecutive_failures = 0
    max_consecutive_failures = 20

    for step in range(1, args.steps + 1):
        if args.synthetic_smoke:
            batch, target_query_pos = make_synthetic_alignment_batch(
                args.batch_size,
                args.n_atoms,
                device,
            )
        elif dataset is not None:
            batch, target_query_pos = sample_training_batch(
                dataset,
                batch_size=args.batch_size,
                conditioning=args.conditioning,
                device=device,
                generator=sample_generator,
            )
            assert target_query_pos is not None
        else:
            batch, target_query_pos, _ = load_alignment_batch_from_pt(
                args.train_data,
                require_target=True,
                device=device,
            )
            assert target_query_pos is not None

        try:
            loss = train_step(
                model=model,
                matcher=matcher,
                optimizer=optimizer,
                batch=batch,
                target_query_pos=target_query_pos,
                lambda_bond=train_config.lambda_bond,
            )
            step_loss = float(loss)
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"Non-finite loss at step={step}: {step_loss}.")
        except FloatingPointError as exc:
            consecutive_failures += 1
            if best_model_state is None or consecutive_failures > max_consecutive_failures:
                raise
            # 중단 대신 복구: 최적 가중치(이미 디스크에 저장됨)로 롤백하고,
            # 옵티마이저 모멘텀을 초기화한 후 학습을 계속한다. 최적 체크포인트는 절대 손실되지 않는다.
            print(
                f"[train] step={step:04d} {exc} "
                f"restoring best@{best_step} (loss={best_loss:.6f}) and continuing."
            )
            model.load_state_dict(best_model_state)
            optimizer.state.clear()
            continue

        consecutive_failures = 0
        final_loss = step_loss
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_step = step
            best_model_state = _clone_state_dict_to_cpu(model)
            # 이후 크래시로 인해 손실되지 않도록 최적 체크포인트를 즉시 저장한다.
            torch.save(
                _checkpoint(best_model_state, {"best_loss": best_loss, "best_step": best_step}),
                best_path,
            )

        if step % args.log_every == 0 or step == 1:
            best_info = f" best={best_loss:.6f}@{best_step}" if best_step is not None else ""
            print(f"[train] step={step:04d} loss={final_loss:.6f}{best_info}")
        
        if val_paths and args.val_every > 0 and step % args.val_every == 0:
            try:
                model.eval()
                val_summary, _ = evaluate_paths(
                    model,
                    val_paths,
                    conditioning=args.conditioning,
                    n_steps=args.val_eval_n_steps,
                    solver=args.val_eval_solver,
                    device=device,
                    limit=args.val_eval_limit,
                )
                print(f"[val]   step={step:04d} {format_summary(val_summary)}")
            except Exception as exc:  # noqa: BLE001 - val monitoring must never kill training
                print(f"[val]   step={step:04d} skipped (eval error: {exc})")
            finally:
                model.train()

    if final_loss is None:
        raise RuntimeError("Training finished without running any optimization step.")

    if best_model_state is None or best_step is None:
        raise RuntimeError(
            "No finite best checkpoint was captured. "
            "Training likely failed before producing a valid loss."
        )

    torch.save(
        _checkpoint(
            _clone_state_dict_to_cpu(model),
            {"final_loss": final_loss, "best_loss": best_loss, "best_step": best_step},
        ),
        save_path,
    )
    torch.save(
        _checkpoint(best_model_state, {"best_loss": best_loss, "best_step": best_step}),
        best_path,
    )

    print(f"[train] checkpoint saved to: {save_path}")
    print(
        f"[train] best checkpoint saved to: {best_path} "
        f"at step={best_step} loss={best_loss:.6f}"
    )

def build_argparser() -> argparse.ArgumentParser:
    """학습 CLI를 정의한다."""
    p = argparse.ArgumentParser(description="Train ETFlowAlign.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument(
        "--source-type",
        type=str,
        default="reference_anchored",
        choices=["gaussian", "reference_anchored", "query_perturbed", "input_query"],
    )
    p.add_argument("--source-noise-scale", type=float, default=0.5)
    p.add_argument("--use-atom-index-embed", action="store_true")
    p.add_argument("--use-direct-vector-head", action="store_true")
    p.add_argument("--max-atoms", type=int, default=256)
    p.add_argument(
        "--use-equivariant-basis-head",
        action="store_true",
        help="Use an equivariant vector basis head instead of the legacy gate*x head.",
    )
    p.add_argument(
        "--use-rigid-head",
        action="store_true",
        help="Direction A: predict a per-graph rigid-body velocity field (preserves geometry).",
    )
    p.add_argument(
        "--use-pocket-conditioning",
        action="store_true",
        help="Feed pocket atoms into the model (ligand<->pocket messages + pocket-shape basis).",
    )
    p.add_argument("--pocket-cutoff", type=float, default=8.0, help="Ligand-pocket edge cutoff (A).")
    p.add_argument("--pocket-max-neighbors", type=int, default=16, help="Max pocket neighbors per ligand atom.")
    p.add_argument(
        "--path-type",
        type=str,
        default="linear",
        choices=["linear", "rigid"],
        help="Flow-matching probability path: linear coordinate interp or SE(3) rigid geodesic.",
    )
    p.add_argument(
        "--use-node-attr",
        action="store_true",
        help="Use query_node_attr/reference_node_attr chemistry features if present.",
    )
    p.add_argument(
        "--node-attr-dim",
        type=int,
        default=5,
        help="Input dimension of node_attr features.",
    )
    p.add_argument(
        "--lambda-bond",
        type=float,
        default=0.0,
        help="Weight for endpoint bond length regularization.",
    )
    p.add_argument("--center-source", dest="center_source", action="store_true")
    p.add_argument("--no-center-source", dest="center_source", action="store_false")
    p.set_defaults(center_source=True)
    p.add_argument("--center-target", dest="center_target", action="store_true")
    p.add_argument("--no-center-target", dest="center_target", action="store_false")
    p.set_defaults(center_target=True)
    p.add_argument("--use-kabsch-alignment", dest="use_kabsch_alignment", action="store_true")
    p.add_argument("--no-use-kabsch-alignment", dest="use_kabsch_alignment", action="store_false")
    p.set_defaults(use_kabsch_alignment=True)
    p.add_argument("--fixed-t", type=float, default=None)

    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-path", type=str, default="etflowalign_ckpt.pt")

    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--train-data", type=str, default="", help="Single .pt batch (overfit/smoke).")
    p.add_argument("--train-dir", type=str, default="", help="Directory of per-complex .pt payloads (dataset training).")
    p.add_argument(
        "--conditioning",
        type=str,
        default="pocket",
        choices=["pocket", "reference", "both"],
        help="Which conditioning signal to keep when training from --train-dir.",
    )
    p.add_argument("--val-fraction", type=float, default=0.0, help="Hold out this fraction of --train-dir as val (never trained on; written to <save-path>.val.txt).")
    p.add_argument("--split-seed", type=int, default=0, help="Seed for the train/val split.")
    p.add_argument("--train-limit", type=int, default=0, help="Use only the first N complexes of --train-dir (0 = all). For overfit diagnostics.") 
    p.add_argument("--val-every", type=int, default=0, help="Evaluate held-out val pose RMSD every N steps (0 = off; needs --val-fraction>0).")
    p.add_argument("--val-eval-limit", type=int, default=100, help="Evaluate at most this many val complexes per check (speed).")
    p.add_argument("--val-eval-n-steps", type=int, default=50, help="ODE steps for val pose sampling.")
    p.add_argument("--val-eval-solver", type=str, default="heun", choices=["euler", "heun"], help="Solver for val pose sampling.")
    return p


def main() -> None:
    """CLI 메인 함수."""
    args = build_argparser().parse_args()
    sources = [bool(args.synthetic_smoke), bool(args.train_data), bool(args.train_dir)]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of --synthetic-smoke, --train-data, or --train-dir.")
    run_training(args)


if __name__ == "__main__":
    main()