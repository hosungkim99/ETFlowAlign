"""강체 모델을 하나의 강체 (소스 -> 타겟) 쌍에 오버피팅시킨 후,
소수 스텝 ODE 추론이 (a) 분자 내부 기하 구조를 보존하고 (b) 타겟 포즈에 수렴하는지 검증한다.
이것은 방향-A의 목표에 대한 직접적인 테스트다.
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
    """단일 강체 쌍에 모델을 오버피팅한 뒤 소수 스텝 ODE 추론을 검증한다.

    한 줄 요약:
        하나의 (소스→타겟) 강체 쌍에 강체 모델을 오버피팅시키고, 여러 솔버/스텝 수로 추론해 기하 보존·수렴을 측정한다.
    생성이유:
        방향-A의 목표(분자 내부 기하를 보존하며 타겟 포즈로 수렴)가 실제 학습-추론
        루프에서 달성되는지 직접 검증하는 sanity test가 필요하다.
    역할:
        유효한 컨포머와 그 강체 변환 타겟을 만들고, flow_matching_step으로 1500스텝 오버피팅한 뒤,
        euler/heun 솔버와 여러 n_steps로 샘플링해 geom_drift/RMSD/COM 거리를 표로 출력한다.
    메커니즘:
        시드 고정 후 conformer/회전 R/평행이동 t로 target과 source를 만들고, 내부 make_batch로
        AlignmentBatch를 구성한다. ETFlowAlignModel(use_rigid_head=True)와 rigid FlowMatcher,
        AdamW로 학습 루프(역전파·grad clip·step)를 돌린다. 평가 단계에서는 ETFlowAlignSampler로
        각 (solver, n_steps) 조합을 샘플링해 쌍거리 변화(geom_drift), 타겟 RMSD, COM 거리를 계산·출력한다.
    파라미터:
        없음.
    반환:
        None: 결과는 학습 손실 로그와 추론 지표 표의 표준출력이다.
    파이프라인 단계:
        4·5단계 검증 - 강체 학습목표/학습루프의 과적합 sanity check.
    """
    torch.manual_seed(0)
    n = 14
    g = torch.Generator().manual_seed(7)
    conformer = torch.randn(n, 3, generator=g) * 1.5          # 유효한 내부 기하 구조
    R = _axis_angle_to_matrix(torch.randn(3, generator=g))
    t = torch.randn(3, generator=g) * 3.0
    target = conformer @ R.transpose(0, 1) + t                 # 강체 변환된 타겟 포즈
    source = conformer.clone()

    atom_type = torch.randint(0, 16, (n,), generator=g)
    zero_batch = torch.zeros(n, dtype=torch.long)

    def make_batch(qpos):
        """주어진 쿼리 좌표로 타겟을 향해 정렬하는 AlignmentBatch를 만든다.

        한 줄 요약:
            qpos를 query_pos로, 외부 target을 reference로 갖는 AlignmentBatch를 생성한다.
        생성이유:
            학습/추론에서 동일한 원자 타입·배치·타겟 레퍼런스를 가진 배치를 반복 생성하기 위한
            클로저로, 바깥의 atom_type/zero_batch/target을 공유한다.
        역할:
            query_pos=qpos, reference_pos=target으로 설정하고 나머지 필드를 공유 값으로 채운 배치를 만든다.
        메커니즘:
            AlignmentBatch를 생성하며 쿼리/레퍼런스 원자 타입과 배치 인덱스에 공유 텐서를 전달한다.
        파라미터:
            qpos (torch.Tensor): 쿼리 좌표 [n,3].
        반환:
            AlignmentBatch: 타겟을 레퍼런스로 갖는 정렬 배치.
        파이프라인 단계:
            4·5단계 검증 - 학습/추론용 배치 구성.
        """
        return AlignmentBatch(
            query_pos=qpos,
            query_atom_type=atom_type,
            query_batch=zero_batch,
            reference_pos=target,                              # 타겟 포즈를 향해 정렬
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
