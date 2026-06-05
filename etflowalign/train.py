"""ETFlowAlign 학습 스크립트 및 유틸리티.

Examples:
    python -m etflowalign.train --synthetic-smoke --steps 2 --batch-size 2 --n-atoms 8 --save-path /tmp/etflowalign_smoke.pt
    python -m etflowalign.train --train-data /path/to/train_batch.pt --steps 1 --save-path /tmp/etflowalign_ckpt.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, optim

from .data import load_alignment_batch_from_pt
from .dataset import AlignmentDataset, sample_training_batch, train_val_split
from .evaluation import evaluate_paths, format_summary
from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class TrainConfig:
    """ETFlowAlign 학습을 위한 최적화 하이퍼파라미터 묶음.

    한 줄 요약:
        옵티마이저(AdamW) 및 손실 가중에 필요한 학습 하이퍼파라미터를 한 곳에 모은 설정 객체.

    생성 이유:
        CLI 인자(argparse)로 흩어져 있는 학습 관련 수치들을 명시적인 타입을 가진 단일
        구조체로 묶어, build_training_components/train_step 같은 하위 함수에 일관된 형태로
        전달하기 위해 만들어졌다. dataclass로 정의해 기본값과 필드 의미를 명확히 한다.

    역할:
        학습 루프(파이프라인 5단계)에서 옵티마이저 생성과 손실 계산에 필요한 스칼라
        하이퍼파라미터의 단일 출처(single source of truth) 역할을 한다.

    정의되는 주요 변수와 의미 (dataclass 필드):
        lr (float): AdamW 옵티마이저의 학습률. 기본값 1e-4.
        weight_decay (float): AdamW의 가중치 감쇠(L2 정규화) 계수. 기본값 0.0(미사용).
        lambda_bond (float): 종단점(endpoint) 결합 길이 정규화 항의 가중치. 0이면 해당
            정규화를 사용하지 않는다. flow_matching_step에 전달되어 손실에 가산된다.

    파이프라인 단계:
        5단계(학습 루프)의 하이퍼파라미터 컨테이너.
    """

    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_bond: float = 0.0


def build_training_components(
    model: ETFlowAlignModel,
    train_config: TrainConfig,
    fm_config: FlowMatchingConfig,
) -> tuple[AlignmentFlowMatcher, optim.Optimizer]:
    """플로우 매처와 옵티마이저를 생성한다.

    한 줄 요약:
        모델과 설정으로부터 flow-matching 학습 목표 객체와 AdamW 옵티마이저를 만들어 반환한다.

    생성 이유:
        학습 루프(run_training)가 직접 매처/옵티마이저 생성 로직을 갖지 않고, 이 헬퍼에
        위임하여 구성 책임을 분리하기 위해 만들어졌다. 테스트나 다른 진입점에서도 동일한
        방식으로 학습 구성요소를 재현할 수 있게 한다.

    역할:
        파이프라인 5단계의 준비 단계에서 학습 목표(flow_matching, 4단계)와 최적화기를
        조립하는 팩토리 역할을 한다.

    메커니즘:
        - fm_config로 AlignmentFlowMatcher를 인스턴스화하여 학습 시 소스/타겟 경로 및
          목표 속도장을 정의한다.
        - model.parameters()와 train_config의 lr/weight_decay로 optim.AdamW를 생성한다.

    파라미터:
        model (ETFlowAlignModel): 학습 대상 모델. 옵티마이저가 추적할 파라미터의 출처.
        train_config (TrainConfig): lr/weight_decay 등 옵티마이저 하이퍼파라미터.
        fm_config (FlowMatchingConfig): flow-matching 경로/노이즈/정렬 관련 설정.

    반환:
        tuple[AlignmentFlowMatcher, optim.Optimizer]:
            (matcher, optimizer). matcher는 학습 목표를 계산하는 객체, optimizer는 모델
            파라미터를 갱신하는 AdamW 인스턴스.

    파이프라인 단계:
        5단계(학습 루프)의 구성요소 빌드.
    """
    matcher = AlignmentFlowMatcher(config=fm_config)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    return matcher, optimizer


def train_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    optimizer: optim.Optimizer,
    batch: AlignmentBatch,
    target_query_pos: Tensor,
    lambda_bond: float = 0.0,
) -> float:
    """최적화 스텝을 한 번 실행하고 스칼라 손실값을 반환한다.

    한 줄 요약:
        한 배치에 대해 순전파→손실 계산→backward→그래디언트 클리핑/정제→optimizer.step을
        수행하는 단일 학습 스텝.

    생성 이유:
        학습 루프 본체(run_training)에서 반복되는 "한 스텝" 로직을 분리하여 재사용 가능하고
        테스트 가능하게 만들기 위해 작성되었다. 특히 NaN/Inf 손실·그래디언트로 인한 가중치
        오염을 방어하는 안전 장치를 한 곳에 모은다.

    역할:
        파이프라인 5단계(학습 루프)의 핵심 연산 단위. 모델 가중치를 실제로 갱신하는 지점.

    메커니즘:
        1) model.train()으로 학습 모드 전환, optimizer.zero_grad로 그래디언트 초기화.
        2) flow_matching_step으로 flow-matching 손실(필요 시 결합 길이 정규화 포함)을 계산.
        3) backward 이전에 손실이 비정상(NaN/Inf)이거나 |loss|>1e4로 폭발하면
           FloatingPointError를 발생시켜 backward/step을 건너뛴다(가중치 보호).
        4) loss.backward() 후 clip_grad_norm_(max_norm=5.0)로 그래디언트 노름을 제한.
        5) clip_grad_norm_이 막지 못하는 NaN/Inf 그래디언트를 nan_to_num_으로 0으로 정제.
        6) optimizer.step()으로 파라미터 갱신, 갱신 후 손실이 비정상이면 다시 예외 발생.

    파라미터:
        model (ETFlowAlignModel): 갱신 대상 모델.
        matcher (AlignmentFlowMatcher): flow-matching 학습 목표 계산기.
        optimizer (optim.Optimizer): 모델 파라미터 갱신용 옵티마이저(AdamW).
        batch (AlignmentBatch): query/reference(및 선택적 pocket) 정보를 담은 입력 배치.
        target_query_pos (Tensor): 정렬 목표가 되는 query 원자의 타겟 좌표 텐서.
        lambda_bond (float): 종단점 결합 길이 정규화 가중치. 0이면 미사용. 기본값 0.0.

    반환:
        float: 해당 스텝의 스칼라 손실값(파이썬 float). CPU로 옮겨 추출한 값.

    예외:
        FloatingPointError: backward 이전 또는 optimizer.step 이후 손실이 비정상/폭발일 때.
            호출자가 이를 잡아 최적 체크포인트로 롤백·복구할 수 있도록 설계되었다.

    파이프라인 단계:
        5단계(학습 루프)의 한 스텝.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = flow_matching_step(
        model,
        matcher,
        batch,
        target_query_pos,
        lambda_bond=lambda_bond,
    )

    loss_check = float(loss.detach())
    if not math.isfinite(loss_check) or abs(loss_check) > 1e4:
        # 비정상(NaN/Inf)이거나 폭발하는 손실값(정상 손실은 ~10-200)을 건너뛴다. 여기서
        # 예외를 발생시키면 backward/step을 우회하므로 하나의 이상 배치가 가중치를 오염시키지
        # 않는다; 호출자는 최적 체크포인트로 롤백하고 학습을 계속한다.
        raise FloatingPointError(f"Non-finite/exploding training loss before backward: {loss_check}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    # 스텝 실행 전에 비정상 그래디언트를 정제한다. clip_grad_norm_은 이를 막지 못한다.
    # NaN/Inf 그래디언트가 하나라도 있으면 total_norm이 NaN이 되어 모든 가중치를 오염시키고,
    # 다음 순전파에서 NaN이 발생해 학습이 종료된다. 잘못된 그래디언트를 0으로 만들면
    # 드물게 발생하는 불안정한 스텝이 크래시 대신 아무 효과 없는 연산으로 처리된다.
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    optimizer.step()

    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            f"Non-finite training loss after optimization step: {loss_value}"
        )

    return loss_value


def make_synthetic_alignment_batch(batch_size: int, n_atoms: int, device: torch.device) -> tuple[AlignmentBatch, Tensor]:
    """스모크 테스트용 간단한 정렬 배치를 생성한다.

    한 줄 요약:
        실제 데이터 없이 학습 파이프라인을 빠르게 검증하기 위한 합성 정렬 배치와 타겟 좌표를 생성.

    생성 이유:
        디스크의 복합체 데이터 없이도 모델/학습 루프가 끝까지 동작하는지 확인하는
        스모크 테스트(--synthetic-smoke)를 위해 만들어졌다. CI나 코드 변경 직후의
        빠른 sanity check 용도.

    역할:
        파이프라인 1·2단계(데이터 준비/로딩)를 대체하는 합성 데이터 생성기. 학습 루프에
        AlignmentBatch와 목표 좌표를 즉석에서 공급한다.

    메커니즘:
        - batch_size개의 그래프 각각에 n_atoms개의 원자를 부여하는 배치 인덱스를 만든다.
        - reference_pos는 가우시안 노이즈(scale 0.7)로 생성하고, atom_type은 0~15 사이 정수.
        - 그래프별 무작위 평행이동(translation)을 reference에 더해 target_query_pos를 만들고,
          약한 사인 비선형(0.05*sin(2x))을 추가해 단순 평행이동이 아니게 만든다.
        - query 초기 좌표(query_init)는 reference에 강한 노이즈(scale 0.8)를 더해, 모델이
          정렬해야 할 흐트러진 시작점을 모사한다.

    파라미터:
        batch_size (int): 배치 내 그래프(복합체) 개수.
        n_atoms (int): 그래프당 원자 개수.
        device (torch.device): 생성된 텐서를 올릴 디바이스.

    반환:
        tuple[AlignmentBatch, Tensor]:
            (batch, target_query_pos). batch는 query/reference 정보를 담은 합성 배치,
            target_query_pos는 query 원자가 도달해야 할 목표 좌표((batch_size*n_atoms, 3)).

    파이프라인 단계:
        5단계(학습 루프)에서 사용되는, 1·2단계 대체용 합성 데이터 생성기.
    """
    query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_atoms)
    reference_batch = query_batch.clone()

    reference_pos = 0.7 * torch.randn(batch_size * n_atoms, 3, device=device)
    atom_type = torch.randint(low=0, high=16, size=(batch_size * n_atoms,), device=device)

    translation = 0.25 * torch.randn(batch_size, 3, device=device)
    target_query_pos = reference_pos + translation[query_batch]
    target_query_pos = target_query_pos + 0.05 * torch.sin(target_query_pos * 2.0)

    query_init = reference_pos + 0.8 * torch.randn_like(reference_pos)
    batch = AlignmentBatch(
        query_pos=query_init,
        query_atom_type=atom_type,
        query_batch=query_batch,
        reference_pos=reference_pos,
        reference_atom_type=atom_type,
        reference_batch=reference_batch,
    )
    return batch, target_query_pos


def _clone_state_dict_to_cpu(model: ETFlowAlignModel) -> dict[str, torch.Tensor]:
    """체크포인트 저장을 위해 모델 파라미터를 CPU로 복제한다.

    한 줄 요약:
        모델 state_dict의 모든 텐서를 그래프에서 분리(detach)·CPU 이전·복제한 새 사전을 반환.

    생성 이유:
        최적/최종 체크포인트를 안전하게 저장하기 위해 만들어졌다. 원본 텐서를 그대로 참조해
        저장하면 이후 학습으로 가중치가 바뀌거나(특히 best 스냅샷이 오염됨), GPU 메모리에
        묶이는 문제가 생긴다. 이를 막기 위해 독립적인 CPU 사본을 만든다.

    역할:
        파이프라인 5단계에서 best/final 체크포인트 스냅샷을 만드는 유틸리티.

    메커니즘:
        model.state_dict()의 각 (key, value)에 대해 value.detach().cpu().clone()을 적용하여,
        자동미분 그래프와 무관하고 디바이스/저장소가 독립적인 텐서 사전을 구성한다.

    파라미터:
        model (ETFlowAlignModel): 파라미터를 복제할 대상 모델.

    반환:
        dict[str, torch.Tensor]: 파라미터 이름 -> CPU 상의 독립 복제 텐서 사전.

    파이프라인 단계:
        5단계(학습 루프)의 체크포인트 저장 보조.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _make_best_checkpoint_path(save_path: str) -> Path:
    """최종 체크포인트 경로로부터 최적 체크포인트 경로를 생성한다.

    한 줄 요약:
        최종 저장 경로를 바탕으로 "_best" 접미사가 붙은 최적 체크포인트 경로를 유도한다.

    생성 이유:
        학습 중 최저 손실 시점의 가중치를 최종 가중치와 별도 파일로 저장하기 위해, 일관된
        규칙으로 best 경로를 만드는 헬퍼가 필요했다.

    역할:
        파이프라인 5단계에서 best 체크포인트 파일 경로를 결정한다.

    메커니즘:
        - save_path의 확장자가 ".pt"이면 "<stem>_best.pt" 형태로 동일 디렉터리에 경로 생성.
        - 그 외 확장자/형식이면 "<save_path>.best.pt"를 새 경로로 사용한다.

    파라미터:
        save_path (str): 최종 체크포인트의 저장 경로 문자열.

    반환:
        Path: 최적 체크포인트를 저장할 파일 경로.

    파이프라인 단계:
        5단계(학습 루프)의 체크포인트 경로 결정.
    """
    path = Path(save_path)
    if path.suffix == ".pt":
        return path.with_name(f"{path.stem}_best{path.suffix}")
    return Path(f"{save_path}.best.pt")


def run_training(args: argparse.Namespace) -> None:
    """CLI 학습 진입점.

    한 줄 요약:
        파싱된 CLI 인자로부터 모델/매처/옵티마이저/데이터를 구성하고 전체 학습 루프를
        실행하며 best·final 체크포인트를 저장하는 최상위 학습 오케스트레이터.

    생성 이유:
        파이프라인 5단계(학습 루프) 전체를 하나의 진입점으로 묶어, main()에서 인자 검증 후
        곧바로 호출할 수 있게 하기 위해 만들어졌다. 데이터 소스 분기, 안정성 복구,
        체크포인트 관리, 주기적 검증 평가를 한 흐름으로 통합한다.

    역할:
        - 모델(ETFlowAlignModel)과 flow-matching 설정(FlowMatchingConfig), 학습 설정
          (TrainConfig)을 인자로부터 구성한다.
        - 데이터 소스를 선택한다: --synthetic-smoke(합성), --train-data(단일 .pt 배치),
          --train-dir(복합체 디렉터리, 선택적 train/val 분할).
        - 매 스텝 배치를 샘플링→train_step으로 갱신→손실 추적→best 체크포인트 즉시 저장.
        - NaN/Inf 등 비정상 손실 시 best 가중치로 롤백·옵티마이저 초기화 후 학습 지속.
        - --val-every 주기로 held-out val에서 evaluate_paths로 포즈 RMSD를 모니터링.
        - 학습 종료 시 final 및 best 체크포인트를 디스크에 저장한다.

    메커니즘:
        - model_args/flow_args/train_args를 dict로 구성해 체크포인트 메타에 함께 저장한다.
        - train_dir 사용 시 AlignmentDataset.from_directory로 경로를 모으고, val_fraction>0
          이면 train_val_split으로 분할 후 val 매니페스트(.val.txt)를 기록한다.
        - 손실이 개선될 때마다 best_loss/best_step/best_model_state를 갱신하고 best 파일을
          즉시 저장하여 크래시에도 최적 가중치가 보존되도록 한다.
        - 연속 실패 횟수가 max_consecutive_failures(20)를 넘거나 best가 없으면 예외를 재발생.

    파라미터:
        args (argparse.Namespace): build_argparser로 정의된 모든 학습 하이퍼파라미터와
            데이터 소스/검증 옵션을 담은 네임스페이스.

    반환:
        None. 부작용으로 체크포인트 파일(best/final)과 선택적 val 매니페스트를 디스크에
        저장하고, 진행 상황을 표준 출력으로 로깅한다.

    예외:
        RuntimeError: 단 한 번의 최적화 스텝도 실행되지 못했거나, 유한한 best 체크포인트를
            한 번도 확보하지 못한 경우.
        FloatingPointError: 복구 한계를 초과한 연속 비정상 손실이 발생한 경우(재발생).

    파이프라인 단계:
        5단계(학습 루프) 전체.
    """
    device = torch.device(args.device)

    model_args = {
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "use_atom_index_embed": args.use_atom_index_embed,
        "use_direct_vector_head": args.use_direct_vector_head,
        "use_equivariant_basis_head": args.use_equivariant_basis_head,
        "use_rigid_head": args.use_rigid_head,
        "use_pocket_conditioning": args.use_pocket_conditioning,
        "pocket_cutoff": args.pocket_cutoff,
        "pocket_max_neighbors": args.pocket_max_neighbors,
        "use_node_attr": args.use_node_attr,
        "node_attr_dim": args.node_attr_dim,
        "max_atoms": args.max_atoms,
    }

    model = ETFlowAlignModel(**model_args).to(device)

    flow_args = {
        "sigma": args.sigma,
        "source_type": args.source_type,
        "source_noise_scale": args.source_noise_scale,
        "center_source": args.center_source,
        "center_target": args.center_target,
        "use_kabsch_alignment": args.use_kabsch_alignment,
        "fixed_t": args.fixed_t,
        "path_type": args.path_type,
    }

    fm_config = FlowMatchingConfig(**flow_args)

    train_config = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, lambda_bond=args.lambda_bond,)
    matcher, optimizer = build_training_components(model, train_config, fm_config)

    train_args = vars(args).copy()

    dataset: AlignmentDataset | None = None
    sample_generator: torch.Generator | None = None
    val_paths: list[str] = []
    if args.train_dir:
        all_paths = AlignmentDataset.from_directory(args.train_dir, require_target=True).paths
        if args.train_limit > 0:
            all_paths = all_paths[: args.train_limit]
        if args.val_fraction > 0.0:
            train_paths, val_paths = train_val_split(all_paths, args.val_fraction, args.split_seed)
            val_manifest = Path(f"{args.save_path}.val.txt")
            val_manifest.parent.mkdir(parents=True, exist_ok=True)
            val_manifest.write_text("\n".join(val_paths))
            dataset = AlignmentDataset(train_paths, require_target=True)
            print(f"[train] held-out val: {len(val_paths)} complexes (never trained) -> {val_manifest}")
        else:
            dataset = AlignmentDataset(all_paths, require_target=True)
        sample_generator = torch.Generator().manual_seed(0)
        print(
            f"[train] dataset: {len(dataset)} train complexes from {args.train_dir} "
            f"| conditioning={args.conditioning} batch_size={args.batch_size}"
        )
    
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = _make_best_checkpoint_path(args.save_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint(model_state: dict[str, torch.Tensor], extra: dict) -> dict:
        """체크포인트 사전을 조립하는 중첩 헬퍼.

        한 줄 요약:
            모델 가중치와 재현에 필요한 인자(model/flow/train)에 추가 메타를 합쳐 저장용
            사전을 만든다.

        생성 이유:
            best와 final 체크포인트를 동일한 구조로 일관되게 저장하기 위해, 공통 필드를
            한 곳에서 구성하도록 run_training 내부에 정의한 클로저다. model_args/flow_args/
            train_args 등 바깥 스코프의 구성 정보를 캡처하여 중복을 줄인다.

        역할:
            torch.save에 전달할 직렬화 가능한 체크포인트 페이로드를 생성한다.

        메커니즘:
            model_state와 model_args/flow_args/train_args를 기본 키로 담은 뒤, extra 사전을
            update로 병합하여 best_loss/best_step/final_loss 같은 상태별 메타를 추가한다.

        파라미터:
            model_state (dict[str, torch.Tensor]): 저장할 모델 파라미터(보통 CPU 복제본).
            extra (dict): best_loss/best_step/final_loss 등 추가 메타데이터.

        반환:
            dict: torch.save로 저장 가능한 체크포인트 사전.

        파이프라인 단계:
            5단계(학습 루프)의 체크포인트 직렬화 보조.
        """
        ckpt = {
            "model_state": model_state,
            "model_args": model_args,
            "flow_args": flow_args,
            "train_args": train_args,
        }
        ckpt.update(extra)
        return ckpt
    
    best_loss = float("inf")
    best_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    final_loss: float | None = None
    consecutive_failures = 0
    max_consecutive_failures = 20

    for step in range(1, args.steps + 1):
        if args.synthetic_smoke:
            batch, target_query_pos = make_synthetic_alignment_batch(
                args.batch_size,
                args.n_atoms,
                device,
            )
        elif dataset is not None:
            batch, target_query_pos = sample_training_batch(
                dataset,
                batch_size=args.batch_size,
                conditioning=args.conditioning,
                device=device,
                generator=sample_generator,
            )
            assert target_query_pos is not None
        else:
            batch, target_query_pos, _ = load_alignment_batch_from_pt(
                args.train_data,
                require_target=True,
                device=device,
            )
            assert target_query_pos is not None

        try:
            loss = train_step(
                model=model,
                matcher=matcher,
                optimizer=optimizer,
                batch=batch,
                target_query_pos=target_query_pos,
                lambda_bond=train_config.lambda_bond,
            )
            step_loss = float(loss)
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"Non-finite loss at step={step}: {step_loss}.")
        except FloatingPointError as exc:
            consecutive_failures += 1
            if best_model_state is None or consecutive_failures > max_consecutive_failures:
                raise
            # 중단 대신 복구: 최적 가중치(이미 디스크에 저장됨)로 롤백하고,
            # 옵티마이저 모멘텀을 초기화한 후 학습을 계속한다. 최적 체크포인트는 절대 손실되지 않는다.
            print(
                f"[train] step={step:04d} {exc} "
                f"restoring best@{best_step} (loss={best_loss:.6f}) and continuing."
            )
            model.load_state_dict(best_model_state)
            optimizer.state.clear()
            continue

        consecutive_failures = 0
        final_loss = step_loss
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_step = step
            best_model_state = _clone_state_dict_to_cpu(model)
            # 이후 크래시로 인해 손실되지 않도록 최적 체크포인트를 즉시 저장한다.
            torch.save(
                _checkpoint(best_model_state, {"best_loss": best_loss, "best_step": best_step}),
                best_path,
            )

        if step % args.log_every == 0 or step == 1:
            best_info = f" best={best_loss:.6f}@{best_step}" if best_step is not None else ""
            print(f"[train] step={step:04d} loss={final_loss:.6f}{best_info}")
        
        if val_paths and args.val_every > 0 and step % args.val_every == 0:
            try:
                model.eval()
                val_summary, _ = evaluate_paths(
                    model,
                    val_paths,
                    conditioning=args.conditioning,
                    n_steps=args.val_eval_n_steps,
                    solver=args.val_eval_solver,
                    device=device,
                    limit=args.val_eval_limit,
                )
                print(f"[val]   step={step:04d} {format_summary(val_summary)}")
            except Exception as exc:  # noqa: BLE001 - val monitoring must never kill training
                print(f"[val]   step={step:04d} skipped (eval error: {exc})")
            finally:
                model.train()

    if final_loss is None:
        raise RuntimeError("Training finished without running any optimization step.")

    if best_model_state is None or best_step is None:
        raise RuntimeError(
            "No finite best checkpoint was captured. "
            "Training likely failed before producing a valid loss."
        )

    torch.save(
        _checkpoint(
            _clone_state_dict_to_cpu(model),
            {"final_loss": final_loss, "best_loss": best_loss, "best_step": best_step},
        ),
        save_path,
    )
    torch.save(
        _checkpoint(best_model_state, {"best_loss": best_loss, "best_step": best_step}),
        best_path,
    )

    print(f"[train] checkpoint saved to: {save_path}")
    print(
        f"[train] best checkpoint saved to: {best_path} "
        f"at step={best_step} loss={best_loss:.6f}"
    )

def build_argparser() -> argparse.ArgumentParser:
    """학습 CLI를 정의한다.

    한 줄 요약:
        ETFlowAlign 학습에 필요한 모든 커맨드라인 인자를 등록한 ArgumentParser를 생성·반환.

    생성 이유:
        모델 구조, flow-matching 경로, 옵티마이저, 데이터 소스, 검증 모니터링 등 다양한
        하이퍼파라미터를 한 곳에서 선언하고, main()에서 파싱해 run_training에 넘기기 위해
        분리된 빌더 함수로 작성되었다. 테스트에서 파서만 단독으로 검증할 수도 있다.

    역할:
        파이프라인 5단계(학습 루프)의 진입 인터페이스(CLI)를 정의한다.

    메커니즘:
        argparse.ArgumentParser에 디바이스/스텝수/배치/모델 차원 등 학습 인자, source_type/
        path_type 등 flow-matching 인자, center/kabsch 관련 store_true·store_false 토글과
        set_defaults, 그리고 데이터 소스(--synthetic-smoke/--train-data/--train-dir)와
        검증 평가(--val-*) 인자를 차례로 add_argument로 등록한다.

    파라미터:
        없음.

    반환:
        argparse.ArgumentParser: 학습 인자가 모두 등록된 파서 인스턴스.

    파이프라인 단계:
        5단계(학습 루프)의 CLI 정의.
    """
    p = argparse.ArgumentParser(description="Train ETFlowAlign.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument(
        "--source-type",
        type=str,
        default="reference_anchored",
        choices=["gaussian", "reference_anchored", "query_perturbed", "input_query"],
    )
    p.add_argument("--source-noise-scale", type=float, default=0.5)
    p.add_argument("--use-atom-index-embed", action="store_true")
    p.add_argument("--use-direct-vector-head", action="store_true")
    p.add_argument("--max-atoms", type=int, default=256)
    p.add_argument(
        "--use-equivariant-basis-head",
        action="store_true",
        help="Use an equivariant vector basis head instead of the legacy gate*x head.",
    )
    p.add_argument(
        "--use-rigid-head",
        action="store_true",
        help="Direction A: predict a per-graph rigid-body velocity field (preserves geometry).",
    )
    p.add_argument(
        "--use-pocket-conditioning",
        action="store_true",
        help="Feed pocket atoms into the model (ligand<->pocket messages + pocket-shape basis).",
    )
    p.add_argument("--pocket-cutoff", type=float, default=8.0, help="Ligand-pocket edge cutoff (A).")
    p.add_argument("--pocket-max-neighbors", type=int, default=16, help="Max pocket neighbors per ligand atom.")
    p.add_argument(
        "--path-type",
        type=str,
        default="linear",
        choices=["linear", "rigid"],
        help="Flow-matching probability path: linear coordinate interp or SE(3) rigid geodesic.",
    )
    p.add_argument(
        "--use-node-attr",
        action="store_true",
        help="Use query_node_attr/reference_node_attr chemistry features if present.",
    )
    p.add_argument(
        "--node-attr-dim",
        type=int,
        default=5,
        help="Input dimension of node_attr features.",
    )
    p.add_argument(
        "--lambda-bond",
        type=float,
        default=0.0,
        help="Weight for endpoint bond length regularization.",
    )
    p.add_argument("--center-source", dest="center_source", action="store_true")
    p.add_argument("--no-center-source", dest="center_source", action="store_false")
    p.set_defaults(center_source=True)
    p.add_argument("--center-target", dest="center_target", action="store_true")
    p.add_argument("--no-center-target", dest="center_target", action="store_false")
    p.set_defaults(center_target=True)
    p.add_argument("--use-kabsch-alignment", dest="use_kabsch_alignment", action="store_true")
    p.add_argument("--no-use-kabsch-alignment", dest="use_kabsch_alignment", action="store_false")
    p.set_defaults(use_kabsch_alignment=True)
    p.add_argument("--fixed-t", type=float, default=None)

    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-path", type=str, default="etflowalign_ckpt.pt")

    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--train-data", type=str, default="", help="Single .pt batch (overfit/smoke).")
    p.add_argument("--train-dir", type=str, default="", help="Directory of per-complex .pt payloads (dataset training).")
    p.add_argument(
        "--conditioning",
        type=str,
        default="pocket",
        choices=["pocket", "reference", "both"],
        help="Which conditioning signal to keep when training from --train-dir.",
    )
    p.add_argument("--val-fraction", type=float, default=0.0, help="Hold out this fraction of --train-dir as val (never trained on; written to <save-path>.val.txt).")
    p.add_argument("--split-seed", type=int, default=0, help="Seed for the train/val split.")
    p.add_argument("--train-limit", type=int, default=0, help="Use only the first N complexes of --train-dir (0 = all). For overfit diagnostics.") 
    p.add_argument("--val-every", type=int, default=0, help="Evaluate held-out val pose RMSD every N steps (0 = off; needs --val-fraction>0).")
    p.add_argument("--val-eval-limit", type=int, default=100, help="Evaluate at most this many val complexes per check (speed).")
    p.add_argument("--val-eval-n-steps", type=int, default=50, help="ODE steps for val pose sampling.")
    p.add_argument("--val-eval-solver", type=str, default="heun", choices=["euler", "heun"], help="Solver for val pose sampling.")
    return p


def main() -> None:
    """CLI 메인 함수.

    한 줄 요약:
        커맨드라인 인자를 파싱·검증하고 run_training을 호출하는 모듈 실행 진입점.

    생성 이유:
        `python -m etflowalign.train` 실행 시 일관된 진입점을 제공하고, 상호 배타적인
        데이터 소스 옵션이 정확히 하나만 지정되었는지 검증하는 책임을 한 곳에 두기 위해
        작성되었다.

    역할:
        파이프라인 5단계(학습 루프)의 최상위 실행 함수. 인자 파싱→유효성 검증→학습 실행을
        연결한다.

    메커니즘:
        - build_argparser().parse_args()로 인자를 파싱한다.
        - --synthetic-smoke / --train-data / --train-dir 중 정확히 하나만 지정되었는지
          확인하고, 그렇지 않으면 ValueError를 발생시킨다.
        - 검증을 통과하면 run_training(args)로 학습을 시작한다.

    파라미터:
        없음(인자는 sys.argv에서 파싱).

    반환:
        None.

    예외:
        ValueError: 데이터 소스 옵션이 정확히 하나가 아닐 때.

    파이프라인 단계:
        5단계(학습 루프)의 진입점.
    """
    args = build_argparser().parse_args()
    sources = [bool(args.synthetic_smoke), bool(args.train_data), bool(args.train_dir)]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of --synthetic-smoke, --train-data, or --train-dir.")
    run_training(args)


if __name__ == "__main__":
    main()