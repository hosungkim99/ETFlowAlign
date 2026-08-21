"""등변성 계약 검증 (SPEC §6) — 백본의 핵심 게이트.

임의 회전 R, 평행이동 τ 에 대해:
  v_θ(R·x + τ)  ==  R · v_θ(x)      (회전 등변)
  v_θ(x + τ)    ==  v_θ(x)          (평행이동 불변)

조건화를 추가할 때마다 이 테스트를 재통과해야 한다.
직접 실행: python -m etflowalign.tests.test_equivariance
"""

from __future__ import annotations

import torch

from etflowalign.backbone import EquivariantTransformer


def _random_rotation(dtype, device):
    """QR 분해로 무작위 특수직교행렬 R (det=+1) 생성."""
    a = torch.randn(3, 3, dtype=dtype, device=device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))          # 부호 고정
    if torch.det(q) < 0:                            # det=+1 보장
        q[:, 0] = -q[:, 0]
    return q


def _make_batch(dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n1, n2 = 12, 9                                   # 두 분자
    z = torch.randint(1, 20, (n1 + n2,), generator=g, device=device)
    pos = torch.randn(n1 + n2, 3, generator=g, device=device, dtype=dtype) * 3.0
    batch = torch.cat([torch.zeros(n1, dtype=torch.long, device=device),
                       torch.ones(n2, dtype=torch.long, device=device)])
    t = torch.rand(2, generator=g, device=device, dtype=dtype)
    return z, pos, t, batch


def check_equivariance(atol: float = 1e-5) -> dict:
    torch.manual_seed(0)
    dtype = torch.float64                            # 엄밀 검증을 위해 double
    device = "cpu"

    model = EquivariantTransformer(hidden_channels=64, num_layers=4, num_rbf=32,
                                   num_heads=4, cutoff=10.0).to(device).to(dtype)
    model.eval()

    # 출력 헤드가 zero-init 이라 초기 출력이 0 -> 등변성이 공허하게 통과함.
    # 실제로 0 이 아닌 출력에서 검증하도록 벡터 경로에 무작위 가중치를 넣는다.
    with torch.no_grad():
        model.velocity_head.vec2_proj.weight.normal_(generator=torch.Generator(device=device).manual_seed(1))

    z, pos, t, batch = _make_batch(dtype, device)
    R = _random_rotation(dtype, device)
    tau = torch.randn(3, dtype=dtype, device=device) * 5.0

    with torch.no_grad():
        v = model(z, pos, t, batch)                          # [N, 3]
        v_rot = model(z, pos @ R.T + tau, t, batch)          # 회전+이동 입력
        v_trans = model(z, pos + tau, t, batch)              # 이동만

    rot_err = (v_rot - v @ R.T).abs().max().item()           # 회전 등변 오차
    trans_err = (v_trans - v).abs().max().item()             # 이동 불변 오차
    out_mag = v.abs().max().item()                           # 출력 크기(공허통과 방지)

    return {
        "rot_equivariance_err": rot_err,
        "trans_invariance_err": trans_err,
        "output_magnitude": out_mag,
        "passed": rot_err < atol and trans_err < atol and out_mag > 1e-3,
        "atol": atol,
    }


def check_conditional_equivariance(atol: float = 1e-5) -> dict:
    """Phase 4: reference 조건화 하에서 joint(query+ref) 회전 등변성."""
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    model = EquivariantTransformer(hidden_channels=64, num_layers=4, num_rbf=32,
                                   num_heads=4, cutoff=10.0).to(device).to(dtype)
    model.eval()
    with torch.no_grad():
        model.velocity_head.vec2_proj.weight.normal_(generator=torch.Generator(device=device).manual_seed(1))

    z, pos, t, batch = _make_batch(dtype, device, seed=0)
    ref_z, ref_pos, _, ref_batch = _make_batch(dtype, device, seed=7)
    R = _random_rotation(dtype, device)
    tau = torch.randn(3, dtype=dtype, device=device) * 5.0

    with torch.no_grad():
        v = model(z, pos, t, batch, ref_z, ref_pos, ref_batch)
        # query 와 reference 를 함께 회전+이동
        v_rot = model(z, pos @ R.T + tau, t, batch,
                      ref_z, ref_pos @ R.T + tau, ref_batch)

    rot_err = (v_rot - v @ R.T).abs().max().item()
    out_mag = v.abs().max().item()
    return {
        "rot_equivariance_err": rot_err,
        "output_magnitude": out_mag,
        "passed": rot_err < atol and out_mag > 1e-3,
        "atol": atol,
    }


def test_equivariance():
    res = check_equivariance()
    assert res["passed"], res


def test_conditional_equivariance():
    res = check_conditional_equivariance()
    assert res["passed"], res


if __name__ == "__main__":
    res = check_equivariance()
    print("--- Phase 3: 조건 없는 백본 ---")
    print(f"[equivariance] rot err   = {res['rot_equivariance_err']:.2e}")
    print(f"[equivariance] trans err = {res['trans_invariance_err']:.2e}")
    print(f"[equivariance] out mag   = {res['output_magnitude']:.3f}  (non-trivial 확인)")
    print(f"[GATE] {'PASS' if res['passed'] else 'FAIL'} (atol={res['atol']:.0e})")

    cres = check_conditional_equivariance()
    print("\n--- Phase 4: reference 조건화 (joint 회전) ---")
    print(f"[equivariance] rot err   = {cres['rot_equivariance_err']:.2e}")
    print(f"[equivariance] out mag   = {cres['output_magnitude']:.3f}")
    print(f"[GATE] {'PASS' if cres['passed'] else 'FAIL'} (atol={cres['atol']:.0e})")
