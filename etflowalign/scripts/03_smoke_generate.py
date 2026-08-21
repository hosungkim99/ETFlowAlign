"""게이트 A 검증: flow-matching 엔진이 유효한 3D 구조를 생성하는가.

실데이터 없이 '장난감 분자'(결합거리 ~1.5Å 무작위 트리)를 소수 만들어
overfit 한 뒤, prior 에서 샘플링해 ODE 로 생성하고:
  - Kabsch-aligned RMSD 가 prior 대비 크게 감소했는가
  - 생성된 결합거리가 정상(~1.5Å)인가
를 본다.

실행: python -m etflowalign.scripts.03_smoke_generate
"""

from __future__ import annotations

import torch

from etflowalign.backbone import EquivariantTransformer
from etflowalign.flow.path import (
    center_pos,
    kabsch_align_source_to_target,
    kabsch_aligned_rmsd,
    raw_rmsd,
)
from etflowalign.flow.prior import get_prior
from etflowalign.sampler import ode_sample
from etflowalign.train import FlowConfig, train_flow


# --------------------------------------------------------------------------
# 장난감 분자: 무작위 스패닝 트리 + 결합거리 1.5Å 랜덤워크 배치
# --------------------------------------------------------------------------
def make_toy_molecule(n_atoms: int, bond_len: float, gen: torch.Generator):
    pos = [torch.zeros(3)]
    edges = []
    for i in range(1, n_atoms):
        parent = int(torch.randint(0, i, (1,), generator=gen).item())
        d = torch.randn(3, generator=gen)
        d = d / d.norm()
        pos.append(pos[parent] + bond_len * d)
        edges.append((parent, i))
    z = torch.randint(1, 10, (n_atoms,), generator=gen)          # 가짜 원자종
    pos = torch.stack(pos, dim=0)
    bonds = torch.tensor(edges, dtype=torch.long).T              # [2, E]
    return z, pos, bonds


def build_batch(num_mols: int, bond_len: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    zs, poss, bonds_list, batch_idx = [], [], [], []
    offset = 0
    for m in range(num_mols):
        n = int(torch.randint(8, 16, (1,), generator=gen).item())
        z, pos, bonds = make_toy_molecule(n, bond_len, gen)
        zs.append(z); poss.append(pos)
        bonds_list.append(bonds + offset)
        batch_idx.append(torch.full((n,), m, dtype=torch.long))
        offset += n
    return {
        "z": torch.cat(zs),
        "pos": torch.cat(poss, dim=0),
        "bonds": torch.cat(bonds_list, dim=1),
        "batch": torch.cat(batch_idx),
    }


def bond_length_stats(pos: torch.Tensor, bonds: torch.Tensor):
    d = (pos[bonds[0]] - pos[bonds[1]]).norm(dim=-1)
    return d.mean().item(), d.std().item()


def per_mol_rmsd(pred, target, batch, metric):
    num = int(batch.max().item()) + 1
    vals = []
    for g in range(num):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        vals.append(metric(pred[idx], target[idx]))
    return sum(vals) / len(vals)


def _to_device(batch: dict, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run(prior_name: str = "gaussian", num_mols: int = 4, steps: int = 3000,
        n_sample_steps: int = 100, bond_len: float = 1.5, seed: int = 0,
        device: str = None, hidden_channels: int = 256, num_layers: int = 8,
        num_rbf: int = 64, num_heads: int = 8):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    torch.manual_seed(seed)
    batch = build_batch(num_mols, bond_len, seed)
    batch["pos"] = center_pos(batch["pos"], batch["batch"])      # 타깃 중심화
    batch = _to_device(batch, device)

    # --- Kabsch 자체검증 (과거 transpose 버그 방지) ---
    Q = batch["pos"][:12]
    R0, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    P = Q @ R0.T
    assert kabsch_aligned_rmsd(P, Q) < 1e-4, "Kabsch 규약 오류"
    # source->target 정렬 자체검증: x1 을 회전시킨 것을 정렬하면 x1 복원
    b0 = torch.zeros(Q.size(0), dtype=torch.long, device=device)
    aligned0 = kabsch_align_source_to_target(P, Q, b0, 1)
    assert kabsch_aligned_rmsd(aligned0, Q) < 1e-4, "source 정렬 오류"
    print("[self-check] Kabsch 회전복원 & source->target 정렬 ~0  OK")

    model = EquivariantTransformer(hidden_channels=hidden_channels, num_layers=num_layers,
                                   num_rbf=num_rbf, num_heads=num_heads, cutoff=10.0).to(device)

    cfg = FlowConfig(prior=prior_name, prior_scale=1.0, lr=2e-4,
                     steps=steps, log_every=max(steps // 10, 1))
    print(f"\n[train] prior={prior_name} mols={num_mols} steps={steps}")
    train_flow(model, [batch], cfg)

    # --- 샘플링 (동일 seed 로 x0 고정하지 않고 prior 에서 생성) ---
    prior = get_prior(prior_name, scale=1.0)
    x0 = prior.sample(batch["z"], batch["bonds"], batch["batch"])
    gen_pos = ode_sample(model, batch["z"], batch["bonds"], batch["batch"],
                         prior, n_steps=n_sample_steps, x0=x0)

    # --- 지표 ---
    target = batch["pos"]
    aligned = per_mol_rmsd(gen_pos, target, batch["batch"], kabsch_aligned_rmsd)
    raw = per_mol_rmsd(gen_pos, target, batch["batch"], raw_rmsd)
    prior_aligned = per_mol_rmsd(x0, target, batch["batch"], kabsch_aligned_rmsd)
    bl_mean, bl_std = bond_length_stats(gen_pos, batch["bonds"])
    tgt_mean, _ = bond_length_stats(target, batch["bonds"])

    print("\n================ GATE A result ================")
    print(f"prior aligned RMSD (baseline) : {prior_aligned:.3f} A")
    print(f"generated aligned RMSD        : {aligned:.3f} A")
    print(f"generated raw RMSD            : {raw:.3f} A")
    print(f"generated bond len mean/std   : {bl_mean:.3f} / {bl_std:.3f} A  (target {tgt_mean:.3f})")

    passed = aligned < 1.0 and 1.3 <= bl_mean <= 1.7
    print(f"\n[GATE A] {'PASS' if passed else 'FAIL'}  "
          f"(criteria: aligned<1.0A, bond len 1.3~1.7A)")
    print("=" * 46)
    return passed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="게이트 A: flow-matching 생성 검증")
    ap.add_argument("--prior", default="harmonic", choices=["gaussian", "harmonic"])
    ap.add_argument("--mols", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--sample-steps", type=int, default=100)
    ap.add_argument("--device", default=None, help="cuda | cpu (기본: 자동)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    args = ap.parse_args()

    run(prior_name=args.prior, num_mols=args.mols, steps=args.steps,
        n_sample_steps=args.sample_steps, device=args.device,
        hidden_channels=args.hidden, num_layers=args.layers)
