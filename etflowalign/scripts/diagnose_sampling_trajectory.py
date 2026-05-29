# etflowalign/scripts/diagnose_sampling_trajectory.py
"""Diagnose bond geometry along ETFlowAlign ODE sampling trajectory.

This script runs ETFlowAlign sampling manually and reports intermediate
geometry statistics at selected ODE steps.

Example:
    python -m etflowalign.scripts.diagnose_sampling_trajectory \
      --checkpoint /path/to/checkpoint_best.pt \
      --input-batch /path/to/diffalign_example_infer.pt \
      --query-sdf /path/to/query.sdf \
      --n-steps 500 \
      --solver euler \
      --report-every 50 \
      --csv-out /path/to/trajectory_bond_diagnostic.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from etflowalign.data import load_alignment_batch_from_pt
from etflowalign.model import AlignmentBatch, ETFlowAlignModel


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose ETFlowAlign sampling trajectory bond geometry."
    )

    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path.")
    parser.add_argument("--input-batch", required=True, help="Inference batch .pt path.")
    parser.add_argument("--query-sdf", required=True, help="Original query SDF path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of ODE steps.")
    parser.add_argument(
        "--solver",
        choices=["euler", "heun"],
        default="euler",
        help="ODE solver.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Report trajectory statistics every N steps.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index label for output. This script diagnoses one deterministic trajectory.",
    )

    return parser


def load_query_reference_geometry(query_sdf: str) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    suppl = Chem.SDMolSupplier(query_sdf, removeHs=False, sanitize=False)
    mol = suppl[0] if len(suppl) > 0 else None
    if mol is None:
        raise RuntimeError(f"Failed to load query SDF: {query_sdf}")

    conf = mol.GetConformer()
    xyz = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        xyz.append([p.x, p.y, p.z])
    xyz_arr = np.asarray(xyz, dtype=float)

    bonds = []
    bond_lengths = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append((i, j))
        bond_lengths.append(np.linalg.norm(xyz_arr[i] - xyz_arr[j]))

    return xyz_arr, bonds, np.asarray(bond_lengths, dtype=float)


def compute_bond_lengths(xyz_arr: np.ndarray, bonds: list[tuple[int, int]]) -> np.ndarray:
    vals = []
    for i, j in bonds:
        vals.append(np.linalg.norm(xyz_arr[i] - xyz_arr[j]))
    return np.asarray(vals, dtype=float)


def geometry_metrics(
    xyz_arr: np.ndarray,
    query_xyz: np.ndarray,
    bonds: list[tuple[int, int]],
) -> dict[str, float]:
    if xyz_arr.shape != query_xyz.shape:
        raise ValueError(f"Shape mismatch: current={xyz_arr.shape}, query={query_xyz.shape}")

    bl = compute_bond_lengths(xyz_arr, bonds)

    rmsd = np.sqrt(((xyz_arr - query_xyz) ** 2).sum(axis=1).mean())
    com_dist = np.linalg.norm(xyz_arr.mean(axis=0) - query_xyz.mean(axis=0))

    return {
        "rmsd_to_query": float(rmsd),
        "com_dist": float(com_dist),
        "bond_min": float(bl.min()),
        "bond_mean": float(bl.mean()),
        "bond_max": float(bl.max()),
    }


def make_step_batch(batch: AlignmentBatch, query_pos: torch.Tensor) -> AlignmentBatch:
    return AlignmentBatch(
        query_pos=query_pos,
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


def restore_global_if_needed(x: torch.Tensor, metadata: dict | None) -> torch.Tensor:
    """Restore global coordinates only for metric comparison with query.sdf.

    export_candidates_to_sdf restores reference_center_subtracted before writing SDF.
    This diagnostic compares intermediate states against query.sdf in global coordinates,
    so it applies the same restoration when metadata has reference_center_subtracted.
    """
    if not isinstance(metadata, dict):
        return x

    center = metadata.get("reference_center_subtracted")
    if center is None:
        return x

    if not torch.is_tensor(center):
        center = torch.as_tensor(center, dtype=x.dtype, device=x.device)
    else:
        center = center.to(device=x.device, dtype=x.dtype)

    if center.ndim == 1 and center.numel() == 3:
        return x + center.view(1, 3)

    return x


def tensor_to_numpy_xyz(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(float)


def assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise FloatingPointError(f"{name} contains non-finite values: count={bad}")


@torch.no_grad()
def run_trajectory(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print("[traj] checkpoint:", args.checkpoint)
    print("[traj] input_batch:", args.input_batch)
    print("[traj] query_sdf:", args.query_sdf)
    print("[traj] device:", device)
    print("[traj] n_steps:", args.n_steps)
    print("[traj] solver:", args.solver)
    print("[traj] report_every:", args.report_every)
    print("[traj] model_args:", ckpt["model_args"])
    print("[traj] flow_args:", ckpt["flow_args"])
    print("[traj] best_loss:", ckpt.get("best_loss"))
    print("[traj] best_step:", ckpt.get("best_step"))

    batch, target_query_pos, metadata = load_alignment_batch_from_pt(
        args.input_batch,
        require_target=False,
        device=device,
    )

    if batch.query_batch.numel() == 0:
        raise ValueError("Empty query batch.")
    num_graphs = int(batch.query_batch.max().item()) + 1
    if num_graphs != 1:
        raise ValueError(f"This simple diagnostic expects one graph, got {num_graphs}")

    query_xyz, bonds, query_bond_lengths = load_query_reference_geometry(args.query_sdf)
    print("[traj] query atoms:", query_xyz.shape[0])
    print("[traj] query bonds:", len(bonds))
    print("[traj] query bond min/mean/max:", query_bond_lengths.min(), query_bond_lengths.mean(), query_bond_lengths.max())

    model = ETFlowAlignModel(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = batch.query_pos.clone()
    assert_finite("initial query_pos", x)

    dt = 1.0 / float(args.n_steps)
    rows = []

    def report(step: int, tval: float, x_current: torch.Tensor) -> None:
        x_global = restore_global_if_needed(x_current, metadata)
        xyz_arr = tensor_to_numpy_xyz(x_global)
        metrics = geometry_metrics(xyz_arr=xyz_arr, query_xyz=query_xyz, bonds=bonds)

        row = {
            "sample_index": args.sample_index,
            "step": step,
            "t": float(tval),
            **metrics,
        }
        rows.append(row)

        print(
            f"step={step:04d} t={tval:.4f} "
            f"rmsd={metrics['rmsd_to_query']:.6f} "
            f"com={metrics['com_dist']:.6f} "
            f"bond_min={metrics['bond_min']:.6f} "
            f"bond_mean={metrics['bond_mean']:.6f} "
            f"bond_max={metrics['bond_max']:.6f}"
        )

    report(step=0, tval=0.0, x_current=x)

    for step in range(args.n_steps):
        t0 = step * dt
        t1 = (step + 1) * dt

        t_graph = torch.full(
            (num_graphs,),
            float(t0),
            device=device,
            dtype=x.dtype,
        )

        step_batch = make_step_batch(batch, x)
        v = model(step_batch, t_graph=t_graph)
        assert_finite("velocity", v)

        if args.solver == "euler":
            x_next = x + dt * v
        elif args.solver == "heun":
            x_euler = x + dt * v
            t_graph_next = torch.full(
                (num_graphs,),
                float(t1),
                device=device,
                dtype=x.dtype,
            )
            next_batch = make_step_batch(batch, x_euler)
            v_next = model(next_batch, t_graph=t_graph_next)
            assert_finite("velocity_next", v_next)
            x_next = x + 0.5 * dt * (v + v_next)
        else:
            raise ValueError(f"Unknown solver: {args.solver}")

        assert_finite("x_next", x_next)
        x = x_next

        if (step + 1) % args.report_every == 0 or (step + 1) == args.n_steps:
            report(step=step + 1, tval=t1, x_current=x)

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("[traj] csv saved to:", out_path)


def main() -> None:
    args = build_argparser().parse_args()
    run_trajectory(args)


if __name__ == "__main__":
    main()