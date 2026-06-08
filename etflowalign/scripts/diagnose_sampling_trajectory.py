# etflowalign/scripts/diagnose_sampling_trajectory.py
"""ETFlowAlign ODE 샘플링 궤적을 따라 결합 기하 구조를 진단한다.

이 스크립트는 ETFlowAlign 샘플링을 수동으로 실행하고,
선택된 ODE 스텝에서의 중간 기하 통계를 출력한다.

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
    """샘플링 궤적 진단 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        이 진단 스크립트에 필요한 명령행 인자를 정의한 ArgumentParser를 만든다.
    생성이유:
        체크포인트/입력 배치/쿼리 SDF 경로와 ODE 설정(스텝 수, 솔버, 보고 주기)을
        사용자가 CLI로 지정할 수 있도록 인자 사양을 한 곳에 모으기 위함.
    역할:
        --checkpoint, --input-batch, --query-sdf, --device, --n-steps, --solver,
        --report-every, --csv-out, --sample-index 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의한다.
        실제 파싱은 수행하지 않고 파서 객체만 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        7단계(추론) 진단 - 샘플링 궤적 점검 도구의 진입점 보조.
    """
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
    """원본 쿼리 SDF에서 기준 좌표와 결합(bond) 기하 구조를 로드한다.

    한 줄 요약:
        query.sdf를 읽어 원자 좌표, 결합 인덱스 목록, 결합 길이를 반환한다.
    생성이유:
        샘플링 궤적의 중간 상태를 비교할 전역 좌표계의 "정답" 기하 구조가 필요하므로,
        원본 쿼리 분자에서 좌표와 결합 길이의 기준값을 추출한다.
    역할:
        RDKit으로 SDF를 로드하고 컨포머 좌표와 모든 결합의 (시작, 끝) 인덱스 및
        각 결합의 길이를 계산해 반환한다.
    메커니즘:
        Chem.SDMolSupplier로 첫 분자를 읽고(수소 유지, sanitize 비활성화),
        컨포머에서 각 원자 위치를 numpy 배열로 모은 뒤, GetBonds로 결합 쌍을 순회하며
        두 원자 좌표 사이의 유클리드 거리를 결합 길이로 계산한다.
    파라미터:
        query_sdf (str): 원본 쿼리 리간드 SDF 파일 경로.
    반환:
        tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
            (좌표 [N,3] float 배열, 결합 인덱스 (i,j) 목록, 결합 길이 [M] float 배열).
    파이프라인 단계:
        7단계(추론) 진단 - 궤적 비교용 기준 기하 구조 준비.
    """
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
    """주어진 좌표와 결합 목록으로부터 결합 길이 배열을 계산한다.

    한 줄 요약:
        좌표 배열과 결합 인덱스로 각 결합의 길이를 계산한다.
    생성이유:
        궤적 중간 상태에서 결합 길이의 보존 여부(기하 왜곡)를 측정하기 위해
        임의 좌표에 대해 반복적으로 결합 길이를 구할 필요가 있다.
    역할:
        bonds의 각 (i, j)에 대해 xyz_arr[i]와 xyz_arr[j] 사이 거리를 계산한다.
    메커니즘:
        결합 쌍을 순회하며 두 좌표 차의 L2 노름을 구해 리스트에 모은 뒤 numpy 배열로 변환한다.
    파라미터:
        xyz_arr (np.ndarray): 원자 좌표 [N,3] 배열.
        bonds (list[tuple[int, int]]): 결합을 이루는 원자 인덱스 (i, j) 목록.
    반환:
        np.ndarray: 각 결합의 길이를 담은 [M] float 배열.
    파이프라인 단계:
        7단계(추론) 진단 - 결합 기하 통계 계산 보조.
    """
    vals = []
    for i, j in bonds:
        vals.append(np.linalg.norm(xyz_arr[i] - xyz_arr[j]))
    return np.asarray(vals, dtype=float)


def geometry_metrics(
    xyz_arr: np.ndarray,
    query_xyz: np.ndarray,
    bonds: list[tuple[int, int]],
) -> dict[str, float]:
    """현재 좌표와 기준 쿼리 좌표를 비교하는 기하 지표를 계산한다.

    한 줄 요약:
        궤적 중간 좌표에 대해 RMSD, COM 거리, 결합 길이 통계를 한 번에 산출한다.
    생성이유:
        샘플링 진행에 따라 포즈가 타겟에 수렴하는지(RMSD, COM)와 내부 결합 기하가
        보존되는지(결합 길이)를 동시에 추적할 단일 지표 집합이 필요하다.
    역할:
        입력 좌표와 기준 좌표의 형태 일치를 확인한 뒤, 원자 단위 RMSD,
        질량중심(COM) 거리, 결합 길이의 최소/평균/최대를 계산한다.
    메커니즘:
        compute_bond_lengths로 결합 길이를 구하고, 좌표 차의 제곱합 평균의 제곱근으로 RMSD를,
        두 좌표 평균(COM) 차의 노름으로 COM 거리를 계산해 딕셔너리로 반환한다.
    파라미터:
        xyz_arr (np.ndarray): 현재(중간) 원자 좌표 [N,3] 배열.
        query_xyz (np.ndarray): 기준 쿼리 원자 좌표 [N,3] 배열.
        bonds (list[tuple[int, int]]): 결합 인덱스 (i, j) 목록.
    반환:
        dict[str, float]: rmsd_to_query, com_dist, bond_min, bond_mean, bond_max 키를 갖는 지표 딕셔너리.
    파이프라인 단계:
        7단계(추론) 진단 - 궤적 스텝별 기하 지표 산출.
    """
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
    """ODE 스텝마다 갱신된 query_pos를 가진 새 AlignmentBatch를 만든다.

    한 줄 요약:
        조건 텐서는 그대로 두고 query_pos만 교체한 AlignmentBatch를 생성한다.
    생성이유:
        ODE 적분 각 스텝에서 모델은 현재 좌표 x_t를 입력으로 받아야 하지만,
        레퍼런스/포켓/노드 속성 등 조건 정보는 변하지 않으므로 좌표만 갈아끼운
        배치를 매 스텝 생성할 필요가 있다.
    역할:
        원본 배치의 모든 조건 필드를 복사하고 query_pos만 새 값으로 설정한다.
    메커니즘:
        AlignmentBatch를 새로 생성하며 query_pos에는 인자로 받은 텐서를,
        나머지 필드(원자 타입, 배치 인덱스, 레퍼런스/포켓, 노드 속성)에는
        기존 batch의 값을 그대로 전달한다.
    파라미터:
        batch (AlignmentBatch): 조건 정보를 담은 원본 배치.
        query_pos (torch.Tensor): 현재 ODE 스텝의 쿼리 좌표 [N,3].
    반환:
        AlignmentBatch: query_pos만 교체되고 조건은 동일한 새 배치.
    파이프라인 단계:
        7단계(추론) 진단 - 스텝별 모델 호출용 배치 구성.
    """
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
    """query.sdf와의 지표 비교를 위해서만 전역 좌표를 복원한다.

    export_candidates_to_sdf는 SDF 저장 전 reference_center_subtracted를 복원한다.
    이 진단 스크립트는 중간 상태를 전역 좌표계의 query.sdf와 비교하므로,
    metadata에 reference_center_subtracted가 있을 경우 동일한 복원 과정을 적용한다.

    한 줄 요약:
        센터링된 좌표에 메타데이터의 중심 벡터를 더해 전역 좌표계로 되돌린다.
    생성이유:
        모델/데이터 파이프라인은 레퍼런스(또는 포켓) 중심을 뺀 센터링 좌표계에서 동작하지만,
        원본 query.sdf는 전역 좌표계에 있어 RMSD 등 비교가 어긋난다. 동일한 중심을 다시
        더해 좌표계를 일치시키기 위해 필요하다.
    역할:
        metadata에 reference_center_subtracted(길이 3 벡터)가 있으면 x에 더해 반환하고,
        없거나 형태가 맞지 않으면 입력을 그대로 반환한다.
    메커니즘:
        metadata가 dict인지 확인하고 center 값을 꺼내, 텐서가 아니면 텐서로 변환하고
        x와 같은 device/dtype으로 맞춘 뒤, 1차원 길이-3 벡터일 때만 [1,3]으로 브로드캐스트해
        x에 더한다.
    파라미터:
        x (torch.Tensor): 센터링된 좌표 [N,3].
        metadata (dict | None): 입력 .pt에서 온 메타데이터(중심 벡터 포함 가능).
    반환:
        torch.Tensor: 중심을 복원한 전역 좌표 또는 변경 없는 입력.
    파이프라인 단계:
        7단계(추론) 진단 - 전역 좌표계 비교를 위한 좌표 복원.
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
    """좌표 텐서를 numpy float 배열로 변환한다.

    한 줄 요약:
        torch 좌표 텐서를 CPU의 float64 numpy 배열로 변환한다.
    생성이유:
        기하 지표 계산(geometry_metrics 등)이 numpy 기반이므로, GPU/그래프에 묶인
        텐서를 안전하게 numpy로 옮길 공통 변환 함수가 필요하다.
    역할:
        텐서를 detach하고 CPU로 옮긴 뒤 numpy로 변환하고 float 타입으로 캐스팅한다.
    메커니즘:
        detach().cpu().numpy().astype(float)를 순차 적용한다.
    파라미터:
        x (torch.Tensor): 변환할 좌표 텐서.
    반환:
        np.ndarray: float 타입 numpy 배열.
    파이프라인 단계:
        7단계(추론) 진단 - 텐서→numpy 변환 보조.
    """
    return x.detach().cpu().numpy().astype(float)


def assert_finite(name: str, x: torch.Tensor) -> None:
    """텐서에 비유한(NaN/Inf) 값이 없는지 검증한다.

    한 줄 요약:
        텐서의 모든 원소가 유한한지 확인하고 아니면 예외를 던진다.
    생성이유:
        ODE 적분 중 발산이나 수치 폭주가 발생하면 조용히 NaN이 전파되므로,
        초기 좌표/속도/갱신값마다 비유한 값을 조기에 잡아내기 위함.
    역할:
        x에 비유한 값이 하나라도 있으면 그 개수와 함께 FloatingPointError를 발생시킨다.
    메커니즘:
        torch.isfinite(x).all()을 검사하고, 실패 시 ~isfinite의 합으로 불량 원소 수를 센 뒤
        name을 포함한 메시지로 예외를 던진다.
    파라미터:
        name (str): 오류 메시지에 표시할 텐서 식별 이름.
        x (torch.Tensor): 검사 대상 텐서.
    반환:
        None: 모두 유한하면 정상 반환, 아니면 예외 발생.
    파이프라인 단계:
        7단계(추론) 진단 - 궤적 수치 안정성 검증.
    """
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise FloatingPointError(f"{name} contains non-finite values: count={bad}")


@torch.no_grad()
def run_trajectory(args: argparse.Namespace) -> None:
    """단일 결정론적 ODE 샘플링 궤적을 실행하며 기하 지표를 보고한다.

    한 줄 요약:
        체크포인트를 불러와 하나의 복합체에 대해 ODE 적분을 수행하고 스텝별 기하 통계를 출력/저장한다.
    생성이유:
        샘플러를 거치지 않고 ODE 적분을 수동으로 풀어, 궤적 중간 단계에서 결합 길이 왜곡이나
        타겟 수렴 양상을 직접 점검하기 위한 진단 절차가 필요하다.
    역할:
        체크포인트/입력 배치/쿼리 SDF를 로드하고 모델을 복원한 뒤, euler/heun 솔버로 n_steps만큼
        적분하면서 report 주기마다 RMSD·COM·결합 길이 통계를 출력하고, 옵션에 따라 CSV로 저장한다.
    메커니즘:
        torch.load로 ckpt 로드 → load_alignment_batch_from_pt로 배치/메타데이터 로드 →
        단일 그래프인지 검증 → load_query_reference_geometry로 기준 기하 확보 →
        ETFlowAlignModel 복원 후 eval. dt=1/n_steps로 루프를 돌며 make_step_batch로 현재 좌표 배치를 만들고
        모델이 속도장 v를 예측, euler(x+dt·v) 또는 heun(예측-보정 평균)으로 좌표를 갱신한다.
        각 단계마다 assert_finite로 수치 안정성을 확인하고, 내부 report 함수로 지표를 기록한다.
    파라미터:
        args (argparse.Namespace): build_argparser가 파싱한 CLI 인자(체크포인트, 입력, 솔버 등).
    반환:
        None: 결과는 표준출력과(지정 시) CSV 파일로 출력된다.
    파이프라인 단계:
        7단계(추론) 진단 - 샘플링 궤적의 기하 안정성/수렴 점검.
    """
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
        """현재 궤적 상태의 기하 지표를 계산해 출력하고 행 목록에 누적한다.

        한 줄 요약:
            주어진 스텝/시간의 좌표를 전역으로 복원해 지표를 계산하고 콘솔 출력 및 CSV용 행으로 저장한다.
        생성이유:
            run_trajectory 내부에서 보고 시점마다 동일한 지표 계산·출력·기록 로직을
            재사용하기 위한 클로저로, 바깥의 metadata/query_xyz/bonds/rows/args를 공유한다.
        역할:
            현재 좌표를 전역 좌표로 복원하고 geometry_metrics로 지표를 산출한 뒤,
            sample_index/step/t와 함께 rows에 추가하고 사람이 읽을 형식으로 출력한다.
        메커니즘:
            restore_global_if_needed → tensor_to_numpy_xyz → geometry_metrics 순으로 처리하고,
            결과 딕셔너리를 rows에 append, 포맷된 문자열을 print한다.
        파라미터:
            step (int): 현재 ODE 스텝 번호.
            tval (float): 현재 flow time t 값.
            x_current (torch.Tensor): 현재 쿼리 좌표 [N,3].
        반환:
            None: 출력과 rows 누적이라는 부수효과만 갖는다.
        파이프라인 단계:
            7단계(추론) 진단 - 궤적 지표 보고.
        """
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
    """CLI 진입점: 인자를 파싱하고 궤적 진단을 실행한다.

    한 줄 요약:
        명령행 인자를 파싱해 run_trajectory를 호출한다.
    생성이유:
        스크립트를 모듈/실행 파일로 호출할 수 있는 표준 진입점을 제공하기 위함.
    역할:
        build_argparser로 인자를 파싱하고 run_trajectory에 전달한다.
    메커니즘:
        build_argparser().parse_args() 결과를 run_trajectory에 넘긴다.
    파라미터:
        없음.
    반환:
        None.
    파이프라인 단계:
        7단계(추론) 진단 - 스크립트 실행 진입점.
    """
    args = build_argparser().parse_args()
    run_trajectory(args)


if __name__ == "__main__":
    main()