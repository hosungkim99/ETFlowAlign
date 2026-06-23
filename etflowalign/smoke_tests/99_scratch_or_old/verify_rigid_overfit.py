"""Overfit a rigid model on one rigid (source -> target) pair, then check that
few-step ODE inference (a) preserves intramolecular geometry and (b) converges to
the target pose. This is the direct test of direction-A's goal.
"""
import torch

from etflowalign.flow_matching import (
    AlignmentFlowMatcher,
    FlowMatchingConfig,
    _axis_angle_to_matrix,
    flow_matching_step,
)
from etflowalign.model import AlignmentBatch, ETFlowAlignModel
from etflowalign.sampler import ETFlowAlignSampler, ODESamplerConfig


def main():
    torch.manual_seed(0)
    n = 14
    g = torch.Generator().manual_seed(7)
    conformer = torch.randn(n, 3, generator=g) * 1.5          # valid internal geometry
    R = _axis_angle_to_matrix(torch.randn(3, generator=g))
    t = torch.randn(3, generator=g) * 3.0
    target = conformer @ R.transpose(0, 1) + t                 # rigid-transformed target pose
    source = conformer.clone()

    atom_type = torch.randint(0, 16, (n,), generator=g)
    zero_batch = torch.zeros(n, dtype=torch.long)

    def make_batch(qpos):
        return AlignmentBatch(
            query_pos=qpos,
            query_atom_type=atom_type,
            query_batch=zero_batch,
            reference_pos=target,                              # align toward target pose
            reference_atom_type=atom_type,
            reference_batch=zero_batch,
        )

    model = ETFlowAlignModel(use_rigid_head=True, hidden_dim=128, num_blocks=4)
    matcher = AlignmentFlowMatcher(
        FlowMatchingConfig(path_type="rigid", source_type="input_query", sigma=0.0)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = make_batch(source)
    for step in range(1, 1501):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = flow_matching_step(model, matcher, batch, target_query_pos=target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 250 == 0:
            print(f"[overfit] step={step} loss={float(loss.detach()):.8f}")

    model.eval()
    src_d = torch.cdist(source, source)
    tgt_com = target.mean(0)
    print(f"\n{'n_steps':>8} {'solver':>6} {'geom_drift':>11} {'rmsd_tgt':>9} {'com_dist':>9}")
    for solver in ("euler", "heun"):
        for n_steps in (3, 5, 10, 50):
            sampler = ETFlowAlignSampler(
                model, ODESamplerConfig(n_steps=n_steps, solver=solver)
            )
            out = sampler.sample(batch=make_batch(source), x0=source.clone())
            geom_drift = float((torch.cdist(out, out) - src_d).abs().max())
            rmsd = float(((out - target) ** 2).sum(-1).mean().sqrt())
            com_dist = float((out.mean(0) - tgt_com).norm())
            print(f"{n_steps:>8} {solver:>6} {geom_drift:>11.4f} {rmsd:>9.4f} {com_dist:>9.4f}")


if __name__ == "__main__":
    main()
