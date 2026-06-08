"""방향-A 강체 경로(rigid path)와 강체 헤드(rigid head)의 수치 검증.

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
    """좌표 집합의 모든 원자 쌍 사이 거리 행렬을 계산한다.

    한 줄 요약:
        좌표 텐서 x에 대해 자기 자신과의 쌍거리 행렬을 반환한다.
    생성이유:
        강체 경로가 내부 기하(원자 간 거리)를 보존하는지 검증하려면 쌍거리 행렬을
        간편하게 구할 보조 함수가 필요하다.
    역할:
        torch.cdist(x, x)로 [N,N] 거리 행렬을 만든다.
    메커니즘:
        torch.cdist를 동일 텐서에 적용한다.
    파라미터:
        x (torch.Tensor): 원자 좌표 [N,3].
    반환:
        torch.Tensor: 쌍거리 행렬 [N,N].
    파이프라인 단계:
        4·5단계 검증 - 기하 보존 측정 보조.
    """
    return torch.cdist(x, x)


def make_rigid_pair(n=12, seed=0):
    """동일 컨포머에서 강체 변환된 (소스, 타겟) 좌표 쌍을 생성한다.

    한 줄 요약:
        무작위 좌표에 무작위 진정 회전과 평행이동을 적용해 (p, q) 강체 쌍을 만든다.
    생성이유:
        강체 확률 경로 검증을 위해, 내부 기하가 동일하면서 강체 변환만 다른
        결정론적 테스트 쌍이 필요하다.
    역할:
        시드로 무작위 좌표 p와 축-각 회전 R, 평행이동 t를 만들고 q = p·Rᵀ + t를 계산한다.
    메커니즘:
        manual_seed로 Generator를 만들고 randn으로 p/omega/t를 생성한다.
        _axis_angle_to_matrix(omega)로 회전 행렬 R(열 벡터 규약)을 만들고, 행 벡터 p에
        Rᵀ를 곱한 뒤 t를 더해 q를 만든다.
    파라미터:
        n (int): 원자 수.
        seed (int): 난수 시드.
    반환:
        tuple[torch.Tensor, torch.Tensor]: (소스 p [n,3], 타겟 q [n,3]).
    파이프라인 단계:
        4·5단계 검증 - 강체 경로 테스트 입력 생성.
    """
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g)
    # 임의의 진정 회전(proper rotation)
    omega = torch.randn(3, generator=g)
    R = _axis_angle_to_matrix(omega)  # 열 벡터 기준(column convention)
    t = torch.randn(3, generator=g) * 2.0
    q = p @ R.transpose(0, 1) + t  # 행 벡터에 열 회전 적용
    return p, q


def test_axis_angle_roundtrip():
    """축-각 ↔ 회전 행렬 변환의 왕복 일관성을 검증한다.

    한 줄 요약:
        무작위 축-각을 회전 행렬로 바꾸고 다시 축-각/행렬로 되돌려 오차가 작은지 확인한다.
    생성이유:
        강체 경로 계산이 의존하는 _axis_angle_to_matrix와 _rotation_to_axis_angle의
        수치 정확성을 보장하기 위함.
    역할:
        여러 무작위 축-각에 대해 R→omega→R 왕복 후 최대 원소 오차를 측정하고 임계값 미만임을 단언한다.
    메커니즘:
        [0,pi) 범위의 omega를 만들어 R을 생성하고, _rotation_to_axis_angle로 omega2를,
        다시 _axis_angle_to_matrix로 R2를 만들어 |R-R2|의 최대값을 누적·출력하고 assert로 검증한다.
    파라미터:
        없음.
    반환:
        None: 검증 통과 시 정상 반환, 실패 시 AssertionError.
    파이프라인 단계:
        4·5단계 검증 - 회전 변환 정확성 sanity check.
    """
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
    """강체 확률 경로의 기하 보존·끝점·속도장 정확성을 검증한다.

    한 줄 요약:
        rigid 경로가 내부 기하를 유지하고 t=0/1에서 소스/타겟에 정확히 닿으며 u_t가 dx/dt와 일치하는지 확인한다.
    생성이유:
        방향-A 강체 경로 구현(_build_rigid_training_state)이 의도한 수학적 성질을 만족하는지
        직접 수치 검증하기 위함.
    역할:
        여러 t에서 보간 상태 x_t와 속도 u_t를 만들어, 쌍거리 보존(geom_err), 끝점 일치(endpoint0/1),
        유한 차분 대비 속도 오차(fd_err)를 측정하고 각각 임계값 미만임을 단언한다.
    메커니즘:
        make_rigid_pair로 (p,q)를 만들고 AlignmentBatch/AlignmentFlowMatcher(rigid)를 구성한다.
        t 목록을 순회하며 _build_rigid_training_state로 (x_t,u_t)를 얻고, _pairwise로 거리 보존을,
        t=0/1에서 끝점 오차를, 내부 t에서는 중심 차분 (x(t+h)-x(t-h))/2h와 u_t의 차이를 측정한다.
    파라미터:
        없음.
    반환:
        None: 검증 통과 시 정상 반환, 실패 시 AssertionError.
    파이프라인 단계:
        4·5단계 검증 - 강체 확률 경로 정확성 검증.
    """
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
        # u_t를 dx_t/dt의 유한 차분으로 검증 (끝점 제외)
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
    """강체 헤드를 사용하는 모델의 순전파가 정상 동작하는지 검증한다.

    한 줄 요약:
        use_rigid_head=True 모델이 [N,3] 유한 출력 속도를 내는지 확인한다.
    생성이유:
        강체 헤드 경로가 형태/수치 측면에서 깨지지 않고 동작하는지 빠르게 점검하기 위함.
    역할:
        무작위 배치에 대해 모델을 호출하고 출력 형태가 (n,3)이며 모두 유한함을 단언한다.
    메커니즘:
        쿼리/레퍼런스를 갖는 AlignmentBatch를 만들고 ETFlowAlignModel(use_rigid_head=True)에 t=0.5로
        순전파한 뒤 v.shape와 torch.isfinite(v).all()을 검증한다.
    파라미터:
        없음.
    반환:
        None: 검증 통과 시 정상 반환, 실패 시 AssertionError.
    파이프라인 단계:
        4·5단계 검증 - 강체 헤드 순전파 sanity check.
    """
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
