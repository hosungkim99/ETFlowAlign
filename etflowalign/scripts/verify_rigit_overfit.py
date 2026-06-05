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
