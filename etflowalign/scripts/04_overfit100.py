"""Phase 5 · overfit-100 (capacity 게이트 B).

실데이터 정렬쌍 100개를 reference 조건화로 '암기'시켜 모델 용량을 확인한다.
못 외우면 일반화는 불가능하므로, 값비싼 일반화 전 반드시 통과해야 하는 관문.

- 조건부 학습(reference) : source 정렬 안 씀(reference 가 프레임 고정).
- harmonic prior 는 PrecomputedHarmonicSampler 로 고유분해 1회만.
- 게이트 B: median Kabsch-aligned RMSD < 2.0A (과거 성공 raw 1.67/aligned 1.29).

서버 실행 예:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python etflowalign/scripts/04_overfit100.py \
      --pt $CODE/geom_pt --n 100 --steps 30000 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.stdout.reconfigure(line_buffering=True)   # 파일 리다이렉트 시에도 줄단위 실시간 출력

from etflowalign.backbone import EquivariantTransformer
from etflowalign.data.pairs import collate
from etflowalign.flow.path import (
    center_by_reference,
    center_pos,
    kabsch_align_source_to_target,
    kabsch_aligned_rmsd,
    raw_rmsd,
)
from etflowalign.flow.prior import PrecomputedHarmonicSampler
from etflowalign.sampler import ode_sample
from etflowalign.train import FlowConfig, train_flow


def load_payloads(pt_dir: str, n: int):
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))[:n]
    if not files:
        raise FileNotFoundError(f".pt 없음: {pt_dir}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def to_device(batch: dict, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_minibatches(payloads, batch_size: int, device):
    """100개를 batch_size 씩 미니배치로. 각 배치에 harmonic 캐시(_prior) 부착."""
    batches = []
    for i in range(0, len(payloads), batch_size):
        b = to_device(collate(payloads[i:i + batch_size]), device)
        b["_prior"] = PrecomputedHarmonicSampler(scale=1.0).build(
            b["z"], b["bonds"], b["batch"])
        batches.append(b)
    return batches


def per_mol_metric(pred, target, batch, metric):
    vals = []
    for g in range(int(batch.max().item()) + 1):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        vals.append(metric(pred[idx], target[idx]))
    return vals


def per_mol_heavy(pred, target, z, batch):
    """heavy-atom(수소 제외) aligned RMSD. DiffAlign/conformer 표준 지표. z>1 원자만."""
    vals = []
    for g in range(int(batch.max().item()) + 1):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        heavy = idx[z[idx] > 1]
        if heavy.numel() >= 3:            # Kabsch 정렬에 최소 3원자
            vals.append(kabsch_aligned_rmsd(pred[heavy], target[heavy]))
    return vals


def run(pt_dir: str, n: int = 100, steps: int = 30000, n_sample_steps: int = 100,
        device: str = None, hidden: int = 256, layers: int = 8, lr: float = 2e-4,
        batch_size: int = 10, save: str = None, force_align: bool = False,
        unconditional: bool = False, load: str = None, n_micro: int = 1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  force_align={force_align}  unconditional={unconditional}  "
          f"sample_steps={n_sample_steps}  n_micro={n_micro}")

    payloads = load_payloads(pt_dir, n)
    batches = make_minibatches(payloads, batch_size, device)
    if unconditional:   # 진단: reference 조건 제거(Gate A 방식, 실분자로)
        for b in batches:
            for k in ("ref_z", "ref_pos", "ref_batch"):
                b.pop(k, None)
    tot_q = sum(int(b["z"].numel()) for b in batches)
    print(f"[data] {len(payloads)} molecules -> {len(batches)} minibatch(size {batch_size}), "
          f"query atoms 총 {tot_q}")

    if load:   # eval-only: 저장 체크포인트 로드, 학습 생략(샘플링만 테스트)
        ckpt = torch.load(load, map_location=device, weights_only=False)
        hidden, layers = ckpt["config"]["hidden"], ckpt["config"]["layers"]
        model = EquivariantTransformer(hidden_channels=hidden, num_layers=layers,
                                       num_rbf=64, num_heads=8, cutoff=10.0).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[load] {load} (hidden={hidden}, layers={layers}) — 학습 생략, 샘플링만")
    else:
        model = EquivariantTransformer(hidden_channels=hidden, num_layers=layers,
                                       num_rbf=64, num_heads=8, cutoff=10.0).to(device)
        # 비조건 모드는 source 정렬 필수(등변). 조건부 진단은 force_align 시에만.
        cfg = FlowConfig(prior="harmonic", lr=lr, steps=steps,
                         log_every=max(steps // 30, 1), use_ema=True,
                         kabsch_source_align=(force_align or unconditional),
                         force_conditioned_align=force_align, n_micro=n_micro)
        print(f"[train] overfit, steps={steps}, minibatch={batch_size}, "
              f"force_align={force_align}")
        train_flow(model, batches, cfg)   # 배치별 _prior 사용

    if save and not load:
        torch.save({"state_dict": model.state_dict(),
                    "config": {"hidden": hidden, "layers": layers}}, save)
        print(f"[save] 체크포인트 저장 -> {save}")

    # 평가: 미니배치별로 중심화 후 샘플 (조건부/비조건 분기)
    aligned, raw, prior_base, heavy = [], [], [], []
    sizes = []   # 분자별 원자 수 (크기-RMSD 상관 진단)
    for b in batches:
        B = b["num_graphs"]
        if "ref_z" in b:   # 조건부: reference COM 중심화 + ref 조건 샘플
            x1c, ref_c = center_by_reference(b["pos"], b["batch"],
                                             b["ref_pos"], b["ref_batch"], B)
            ref_kwargs = dict(ref_z=b["ref_z"], ref_pos=ref_c, ref_batch=b["ref_batch"])
        else:              # 비조건: 단순 중심화
            x1c = center_pos(b["pos"], b["batch"])
            ref_kwargs = {}
        x0 = b["_prior"].sample(b["z"], b["bonds"], b["batch"])   # 시작점
        prior_base += per_mol_metric(x0, x1c, b["batch"], kabsch_aligned_rmsd)
        if force_align or unconditional:   # 회전 제거하고 시작
            x0 = kabsch_align_source_to_target(x0, x1c, b["batch"], B)
        gen = ode_sample(model, b["z"], b["bonds"], b["batch"], b["_prior"],
                         n_steps=n_sample_steps, x0=x0, **ref_kwargs)
        aligned += per_mol_metric(gen, x1c, b["batch"], kabsch_aligned_rmsd)
        raw += per_mol_metric(gen, x1c, b["batch"], raw_rmsd)
        heavy += per_mol_heavy(gen, x1c, b["z"], b["batch"])   # 수소 제외 RMSD
        for g in range(B):
            sizes.append(int((b["batch"] == g).sum().item()))

    # 크기-RMSD 상관 진단: 작은 분자 vs 큰 분자 (cutoff 전제 검증)
    pairs = sorted(zip(sizes, [a for a in aligned]))  # aligned 는 배치순=수집순
    half = len(pairs) // 2
    small = [r for _, r in pairs[:half]]
    large = [r for _, r in pairs[half:]]
    print("\n[크기 진단] 분자당 (원자수, aligned RMSD), 크기순:")
    for sz, r in pairs:
        print(f"    natoms={sz:3d}  aligned={r:.2f}")
    if small and large:
        print(f"[크기 진단] 작은절반 평균={sum(small)/len(small):.3f}  "
              f"큰절반 평균={sum(large)/len(large):.3f}  "
              f"(큰쪽이 훨씬 크면 cutoff 문제 유력)")
    aligned.sort(); raw.sort(); prior_base.sort(); heavy.sort()
    med_a = aligned[len(aligned) // 2]
    med_r = raw[len(raw) // 2]
    med_p = prior_base[len(prior_base) // 2]
    med_h = heavy[len(heavy) // 2] if heavy else float("nan")

    # 원자 조성 (H 포함 여부 확인 — heavy-atom 지표가 의미있는지)
    all_z = torch.cat([b["z"] for b in batches])
    n_h = int((all_z == 1).sum()); n_heavy = int((all_z > 1).sum())

    print("\n================ GATE B (overfit-100) ================")
    print(f"원자 조성: heavy={n_heavy}, H={n_h}  (H 비율 {n_h/max(n_heavy+n_h,1)*100:.0f}%)")
    print(f"prior 기준선 aligned : {med_p:.3f} A")
    print(f"median aligned RMSD (all-atom) : {med_a:.3f} A")
    print(f"median heavy-atom RMSD        : {med_h:.3f} A   <- 표준 지표")
    print(f"median raw RMSD               : {med_r:.3f} A")
    print(f"all-atom <2A : {sum(a < 2.0 for a in aligned)}/{len(aligned)}   "
          f"heavy <2A : {sum(a < 2.0 for a in heavy)}/{len(heavy)}   "
          f"heavy <1A : {sum(a < 1.0 for a in heavy)}/{len(heavy)}")
    passed = med_h < 2.0
    print(f"\n[GATE B] {'PASS' if passed else 'FAIL'}  (median heavy-atom < 2.0A)")
    print("=" * 54)
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help=".pt 디렉터리 (geom_pt)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--sample-steps", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=10, help="미니배치 분자 수(OOM시 낮춤)")
    ap.add_argument("--save", default=None, help="학습 후 체크포인트 저장 경로")
    ap.add_argument("--force-align", action="store_true",
                    help="진단: 조건부에서도 source 정렬 강제(회전 제거 시 암기 가능한가)")
    ap.add_argument("--unconditional", action="store_true",
                    help="진단: reference 조건 제거(실분자를 Gate A 방식으로 외우나)")
    ap.add_argument("--load", default=None,
                    help="eval-only: 체크포인트 로드 후 학습 생략, 샘플링만(샘플스텝 테스트)")
    ap.add_argument("--micro", type=int, default=1,
                    help="스텝당 (x0,t) 샘플 수(K>1=분산↓, B1 진단)")
    args = ap.parse_args()
    run(pt_dir=args.pt, n=args.n, steps=args.steps, n_sample_steps=args.sample_steps,
        device=args.device, hidden=args.hidden, layers=args.layers, lr=args.lr,
        batch_size=args.batch_size, save=args.save, force_align=args.force_align,
        unconditional=args.unconditional, load=args.load, n_micro=args.micro)
