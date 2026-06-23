"""Numeric verification of the direction-A rigid path and rigid head.

Run: python -m etflowalign.smoke_tests.99_scratch_or_old.verify_rigid_path
"""
import torch

from etflowalign.flow_matching import (
    AlignmentFlowMatcher,
    FlowMatchingConfig,
    _axis_angle_to_matrix,
    _rotation_to_axis_angle,
)
from etflowalign.model import AlignmentBatch, ETFlowAlignModel


def _pairwise(x):
    return torch.cdist(x, x)


def make_rigid_pair(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g)
    # random proper rotation
    omega = torch.randn(3, generator=g)
    R = _axis_angle_to_matrix(omega)  # column convention
    t = torch.randn(3, generator=g) * 2.0
    q = p @ R.transpose(0, 1) + t  # apply column rotation to rows
    return p, q


def test_axis_angle_roundtrip():
    g = torch.Generator().manual_seed(3)
    max_err = 0.0
    for _ in range(200):
        omega = torch.randn(3, generator=g)
        omega = omega / omega.norm() * (torch.rand(1, generator=g) * 3.1)  # angle in [0,pi)
        R = _axis_angle_to_matrix(omega)
        omega2 = _rotation_to_axis_angle(R)
        R2 = _axis_angle_to_matrix(omega2)
        max_err = max(max_err, float((R - R2).abs().max()))
    print(f"[axis-angle roundtrip] max |R-R2| = {max_err:.3e}")
    assert max_err < 1e-4, max_err


def test_rigid_path():
    p, q = make_rigid_pair(n=15, seed=1)
    n = p.size(0)
    batch = AlignmentBatch(
        query_pos=p,
        query_atom_type=torch.zeros(n, dtype=torch.long),
        query_batch=torch.zeros(n, dtype=torch.long),
    )
    matcher = AlignmentFlowMatcher(
        FlowMatchingConfig(path_type="rigid", source_type="input_query", sigma=0.0)
    )

    src_d = _pairwise(p)
    geom_err = 0.0
    endpoint0 = endpoint1 = 0.0
    fd_err = 0.0
    for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        t_node = torch.full((n,), float(t))
        x_t, u_t = matcher._build_rigid_training_state(batch, p, q, t_node)
        geom_err = max(geom_err, float((_pairwise(x_t) - src_d).abs().max()))
        if t == 0.0:
            endpoint0 = float((x_t - p).abs().max())
        if t == 1.0:
            endpoint1 = float((x_t - q).abs().max())
        # finite-difference check of u_t against dx_t/dt (skip endpoints)
        if 0.0 < t < 1.0:
            h = 1e-4
            xp, _ = matcher._build_rigid_training_state(batch, p, q, torch.full((n,), t + h))
            xm, _ = matcher._build_rigid_training_state(batch, p, q, torch.full((n,), t - h))
            fd = (xp - xm) / (2 * h)
            fd_err = max(fd_err, float((fd - u_t).abs().max()))

    print(f"[rigid path] geometry drift (pairwise dist)   = {geom_err:.3e}")
    print(f"[rigid path] endpoint t=0 |x_t - x0|          = {endpoint0:.3e}")
    print(f"[rigid path] endpoint t=1 |x_t - x1| (exact)  = {endpoint1:.3e}")
    print(f"[rigid path] u_t vs finite-diff dx/dt         = {fd_err:.3e}")
    assert geom_err < 1e-4
    assert endpoint0 < 1e-5
    assert endpoint1 < 1e-4
    assert fd_err < 1e-2


def test_rigid_head_runs():
    n = 20
    batch = AlignmentBatch(
        query_pos=torch.randn(n, 3),
        query_atom_type=torch.randint(0, 16, (n,)),
        query_batch=torch.zeros(n, dtype=torch.long),
        reference_pos=torch.randn(n, 3),
        reference_atom_type=torch.randint(0, 16, (n,)),
        reference_batch=torch.zeros(n, dtype=torch.long),
    )
    model = ETFlowAlignModel(use_rigid_head=True)
    v = model(batch, t_graph=torch.tensor([0.5]))
    print(f"[rigid head] output shape={tuple(v.shape)} finite={bool(torch.isfinite(v).all())}")
    assert v.shape == (n, 3)
    assert torch.isfinite(v).all()


if __name__ == "__main__":
    test_axis_angle_roundtrip()
    test_rigid_path()
    test_rigid_head_runs()
    print("ALL RIGID CHECKS PASSED")
