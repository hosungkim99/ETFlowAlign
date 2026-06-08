"""ETFlowAlign을 위한 다중 예제 데이터셋 및 콜레이션.

데이터셋은 복합체별 ``.pt`` 페이로드의 집합이다(``diffalign_adapter`` / PDBbind
추출기가 생성하는 동일한 스키마): 각 파일은 ``query_pos``, ``query_atom_type``,
``target_query_pos`` 및 선택적 ``reference_*`` / ``pocket_*`` / 결합 필드를
포함하는 하나의 복합체를 담는다.

이 모듈은 단일 복합체 페이로드를 모델이 이미 이해하는 배치형 멀티-그래프
``AlignmentBatch`` 객체로 변환하며, 동일한 데이터를 포켓 전용, 레퍼런스 전용,
또는 둘 다로 학습할 수 있는 컨디셔닝 스위치를 제공한다.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from typing import Literal, Optional

import torch
from torch import Tensor

from .data import load_alignment_batch_from_pt
from .model import AlignmentBatch

Conditioning = Literal["pocket", "reference", "both"]


class AlignmentDataset:
    """복합체별 ``.pt`` 페이로드 경로 목록에 대한 지연 로딩 데이터셋.

    한 줄 요약:
        복합체별 ``.pt`` 파일 경로 목록을 인덱싱 가능한 데이터셋으로 감싸,
        접근 시점에 비로소 디스크에서 ``AlignmentBatch``를 로드한다.

    생성 이유:
        모든 복합체를 메모리에 미리 적재하면 대규모 데이터셋에서 비효율적이므로,
        경로만 보관하고 ``__getitem__`` 호출 때마다 해당 파일을 읽는 지연 로딩
        방식이 필요하다. 이를 통해 ``sample_training_batch``가 임의의 인덱스
        부분집합만 로드해 배치를 구성할 수 있다.

    역할:
        파이프라인 2단계(데이터 로딩/배치 구성)에서 학습/검증용 복합체 컬렉션을
        표현하며, 인덱스를 단일 복합체 ``(AlignmentBatch, target)`` 쌍으로 변환한다.

    정의되는 주요 변수와 의미:
        paths (list[str]): 로드 대상 ``.pt`` 파일들의 경로 목록.
        require_target (bool): 각 파일에 정답 좌표(``target_query_pos``)가
            반드시 존재해야 하는지 여부. 학습에는 True, 추론에는 False가 일반적.
    """

    def __init__(self, paths: list[str], require_target: bool = True) -> None:
        """데이터셋 인스턴스를 초기화한다.

        한 줄 요약:
            경로 목록과 정답 요구 여부를 받아 데이터셋 속성으로 저장한다.

        역할:
            빈 경로 목록을 조기에 거부하고, 경로 목록의 복사본과 정답 요구
            플래그를 인스턴스 속성으로 고정한다.

        메커니즘:
            ``paths``를 ``list(...)``로 복제해 외부 변경으로부터 격리하고,
            ``require_target``을 ``bool()``로 정규화한다. 경로가 비어 있으면
            ``ValueError``를 던진다.

        파라미터:
            paths (list[str]): 복합체별 ``.pt`` 파일 경로 목록. 비어 있으면 안 된다.
            require_target (bool): 정답 좌표 존재를 강제할지 여부(기본 True).

        생성 속성:
            self.paths (list[str]): 보관된 경로 목록의 복사본.
            self.require_target (bool): 정규화된 정답 요구 플래그.

        파이프라인 단계:
            2단계(데이터 로딩/배치 구성)의 데이터셋 초기화.
        """
        if not paths:
            raise ValueError("AlignmentDataset received an empty path list.")
        self.paths = list(paths)
        self.require_target = bool(require_target)

    def __len__(self) -> int:
        """데이터셋에 포함된 복합체(파일) 개수를 반환한다.

        한 줄 요약:
            보관된 경로 개수를 데이터셋 크기로 돌려준다.

        생성 이유:
            ``torch.randperm`` 등 인덱싱 기반 샘플링이 데이터셋 크기를
            알아야 하므로 표준 ``__len__`` 프로토콜을 제공한다.

        역할:
            데이터셋의 예제 총 개수를 노출한다.

        메커니즘:
            ``len(self.paths)``를 그대로 반환한다.

        파라미터:
            없음.

        반환:
            int: 데이터셋에 들어 있는 복합체 파일 수.

        파이프라인 단계:
            2단계(데이터 로딩/배치 구성)의 데이터셋 크기 조회.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[AlignmentBatch, Optional[Tensor]]:
        """인덱스에 해당하는 복합체를 로드해 ``(배치, 정답)`` 쌍으로 반환한다.

        한 줄 요약:
            ``index`` 위치의 ``.pt`` 파일을 읽어 단일 복합체 ``AlignmentBatch``와
            정답 좌표를 반환한다.

        생성 이유:
            지연 로딩 데이터셋이 개별 예제를 표준 인덱싱 구문으로 제공하도록
            하기 위해 필요하다. 콜레이션 단계가 여러 예제를 모아 배치를 만든다.

        역할:
            경로 -> 단일 복합체 예제 변환을 수행하되, 디바이스 이동은 하지 않고
            CPU 텐서 상태로 반환한다(콜레이션/디바이스 이동은 이후 단계 담당).

        메커니즘:
            ``load_alignment_batch_from_pt``를 ``require_target=self.require_target``,
            ``device=None``으로 호출하고, 반환된 메타데이터는 버린다.

        파라미터:
            index (int): 로드할 복합체의 위치 인덱스(0..len-1).

        반환:
            tuple[AlignmentBatch, Optional[Tensor]]: 단일 복합체 배치와 정답
                좌표 텐서(``target_query_pos``). 정답이 없으면 두 번째 원소는 None.

        파이프라인 단계:
            2단계(데이터 로딩/배치 구성)의 단일 예제 로딩.
        """
        batch, target, _ = load_alignment_batch_from_pt(
            self.paths[index],
            require_target=self.require_target,
            device=None,  # 콜레이션/디바이스 이동은 나중에 수행된다
        )
        return batch, target

    @classmethod
    def from_directory(cls, root: str, pattern: str = "*.pt", require_target: bool = True) -> "AlignmentDataset":
        """디렉터리를 재귀 탐색해 일치하는 파일들로 데이터셋을 생성한다.

        한 줄 요약:
            ``root`` 아래에서 ``pattern``에 맞는 ``.pt`` 파일을 재귀 수집해
            ``AlignmentDataset``을 만든다.

        생성 이유:
            준비 스크립트(``scripts/prepare_*.py``)가 생성한 복합체 파일들이
            보통 한 디렉터리 트리에 흩어져 저장되므로, 경로를 일일이 나열하지
            않고 디렉터리 단위로 데이터셋을 구성하는 편의 생성자가 필요하다.

        역할:
            파이프라인 2단계에서 디스크 디렉터리 -> 데이터셋 인스턴스 변환을 담당.

        메커니즘:
            ``glob.glob(os.path.join(root, "**", pattern), recursive=True)``로
            하위 디렉터리까지 탐색하고, 결정론적 순서를 위해 ``sorted``로 정렬한
            뒤 클래스 생성자에 전달한다.

        파라미터:
            root (str): 탐색을 시작할 루트 디렉터리 경로.
            pattern (str): 파일 매칭 glob 패턴(기본 ``"*.pt"``).
            require_target (bool): 생성된 데이터셋의 정답 요구 플래그(기본 True).

        반환:
            AlignmentDataset: 매칭된 경로들로 구성된 데이터셋 인스턴스.

        파이프라인 단계:
            2단계(데이터 로딩/배치 구성)의 데이터셋 일괄 구성.
        """
        paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        return cls(paths, require_target=require_target)


def train_val_split(
    paths: list[str],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """페이로드 경로를 (train, val)로 결정론적 무작위 분할한다.

    한 줄 요약:
        경로 목록을 시드 고정 난수로 섞어 검증 비율만큼 떼어 학습/검증 경로
        목록으로 나눈다.

    생성 이유:
        학습/검증 분할이 실행마다 달라지면 실험 재현이 불가능하므로, 시드로
        고정된 결정론적 분할이 필요하다. 또한 입력 순서에 무관하게 동일한
        분할을 얻기 위해 정렬 후 섞는다.

    역할:
        파이프라인 2단계에서 전체 복합체 경로 집합을 학습/검증 부분집합으로
        나누어 ``AlignmentDataset`` 두 개를 구성할 근거를 제공한다.

    메커니즘:
        ``sorted``로 입력 순서를 정규화한 뒤, ``seed``로 초기화한
        ``torch.Generator``로 ``randperm`` 순열을 생성한다. 순열 앞쪽
        ``n_val = round(N * val_fraction)``개 인덱스를 검증 집합으로 삼고
        나머지를 학습 집합으로 분류하되, 각 목록은 원래 정렬 순서를 유지한다.

    파라미터:
        paths (list[str]): 분할 대상 ``.pt`` 경로 목록.
        val_fraction (float): 검증 집합 비율, ``[0, 1)`` 범위(기본 0.1).
        seed (int): 분할을 재현 가능하게 만드는 난수 시드(기본 0).

    반환:
        tuple[list[str], list[str]]: ``(train_paths, val_paths)`` 튜플.

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 학습/검증 분할.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    ordered = sorted(paths)
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(ordered), generator=generator).tolist()
    n_val = int(round(len(ordered) * val_fraction))
    val_idx = set(perm[:n_val])
    train = [ordered[i] for i in range(len(ordered)) if i not in val_idx]
    val = [ordered[i] for i in range(len(ordered)) if i in val_idx]
    return train, val


def apply_conditioning(batch: AlignmentBatch, conditioning: Conditioning) -> AlignmentBatch:
    """모델이 선택된 신호로 학습하도록 reference 및/또는 pocket 필드를 마스킹한다.

    한 줄 요약:
        컨디셔닝 모드에 따라 배치에서 reference 또는 pocket 관련 필드를 None으로
        비워 선택된 조건 신호만 모델에 노출한다.

    생성 이유:
        동일한 데이터로 포켓 전용 / 레퍼런스 전용 / 둘 다 컨디셔닝을 학습·평가할
        수 있어야 하므로, 불필요한 조건 신호를 제거하는 스위치가 필요하다.

    역할:
        파이프라인 2단계에서 콜레이션된 배치에 컨디셔닝 마스킹을 적용해,
        모델이 의도한 조건 신호만 입력으로 받도록 한다.

    메커니즘:
        ``"pocket"``이면 reference 관련 4개 필드를, ``"reference"``이면 pocket
        관련 3개 필드를 None으로 지정한 dict를 만들어 ``dataclasses.replace``로
        새 배치를 만든다. ``"both"``는 원본 배치를 그대로 반환하고, 그 외 값은
        ``ValueError``를 던진다. 원본 배치는 변경하지 않는다(불변 처리).

    파라미터:
        batch (AlignmentBatch): 컨디셔닝을 적용할 입력 배치.
        conditioning (Conditioning): ``"pocket"`` / ``"reference"`` / ``"both"``
            중 하나의 컨디셔닝 모드.

    반환:
        AlignmentBatch: 선택된 조건 필드만 남기고 나머지를 비운 새 배치
            (``"both"``는 입력 배치 그대로).

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 conditioning 선택.
    """
    drop: dict[str, None] = {}
    if conditioning == "pocket":
        drop = {"reference_pos": None, "reference_atom_type": None, "reference_batch": None, "reference_node_attr": None}
    elif conditioning == "reference":
        drop = {"pocket_pos": None, "pocket_batch": None, "pocket_atom_type": None}
    elif conditioning == "both":
        return batch
    else:
        raise ValueError(f"Unknown conditioning: {conditioning!r}")
    return dataclasses.replace(batch, **drop)


def _all_present(items: list[Optional[Tensor]]) -> bool:
    """리스트의 모든 텐서가 존재하며 비어 있지 않은지 검사한다.

    한 줄 요약:
        텐서 리스트의 모든 원소가 None이 아니고 원소 수가 0보다 큰지 판정한다.

    생성 이유:
        콜레이션 시 어떤 그래프는 reference/pocket/bond 등 선택적 필드를 갖지
        않을 수 있다. 모든 그래프가 해당 필드를 갖출 때만 이어 붙이도록 일괄
        존재 여부를 확인하는 가드가 필요하다.

    역할:
        콜레이션 단계에서 선택적 필드를 합칠지(전부 존재) 또는 None으로 둘지
        결정하는 boolean 게이트 역할.

    메커니즘:
        각 원소에 대해 ``x is not None``과 ``x.numel() > 0``을 모두 만족하는지
        ``all(...)``로 검사한다.

    파라미터:
        items (list[Optional[Tensor]]): 검사 대상 텐서(또는 None) 리스트.

    반환:
        bool: 모든 원소가 존재하고 비어 있지 않으면 True, 아니면 False.

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 콜레이션 보조.
    """
    return all(x is not None and x.numel() > 0 for x in items)


def collate_alignment(
    items: list[tuple[AlignmentBatch, Optional[Tensor]]],
    conditioning: Conditioning = "both",
    device: str | torch.device | None = None,
) -> tuple[AlignmentBatch, Optional[Tensor]]:
    """단일 복합체 예제들을 하나의 멀티-그래프 ``AlignmentBatch``로 합친다.

    각 입력 예제는 그래프 ``g`` (0..B-1)로 취급된다. 노드 텐서는 이어 붙이고;
    ``*_batch`` 인덱스와 ``query_bond_index``는 병합된 배치가 내부적으로
    일관성을 유지하도록 오프셋이 재조정된다.

    한 줄 요약:
        여러 개의 단일 복합체 ``(AlignmentBatch, target)`` 예제를 하나의 배치형
        멀티-그래프 ``AlignmentBatch``와 결합 정답 텐서로 콜레이션한다.

    생성 이유:
        모델은 여러 복합체를 한 번에 처리하는 멀티-그래프 배치를 입력으로
        받으므로, 개별 예제의 노드/엣지 텐서를 이어 붙이고 그래프 인덱스를
        재매핑하는 콜레이션 로직이 필요하다.

    역할:
        파이프라인 2단계에서 데이터셋이 내준 예제 리스트를 모델이 이해하는
        단일 배치로 변환하고, 컨디셔닝 마스킹과 선택적 디바이스 이동까지 적용한다.

    메커니즘:
        query 노드 텐서(``query_pos``/``query_atom_type``)를 dim=0으로 이어 붙이고
        각 그래프에 0..B-1 인덱스를 부여한 ``query_batch``를 만든다. 결합 인덱스
        재매핑용 그래프별 누적 원자 오프셋(``atom_offsets``)을 계산한다. 내부
        헬퍼 ``cat_node``/``cat_with_batch``로 선택적 노드 필드(reference/pocket)와
        그에 대응하는 batch 인덱스를 합치되, 모든 그래프에 존재할 때만 합치고
        아니면 None으로 둔다. 결합(bond)은 그래프별 원자 오프셋을 더한 뒤
        dim=1로 이어 붙이고 결합 길이는 dim=0으로 이어 붙인다. 합쳐진 필드로
        ``AlignmentBatch``를 만든 뒤 ``apply_conditioning``으로 마스킹하고,
        정답이 모두 존재하면 dim=0으로 이어 붙인다. ``device``가 주어지면
        ``_batch_to_device``로 배치와 정답을 이동한다.

    파라미터:
        items (list[tuple[AlignmentBatch, Optional[Tensor]]]): 콜레이션할 단일
            복합체 ``(배치, 정답)`` 쌍의 리스트. 비어 있으면 안 된다.
        conditioning (Conditioning): 병합 배치에 적용할 컨디셔닝 모드
            (기본 ``"both"``).
        device (str | torch.device | None): 결과를 이동할 디바이스. None이면
            CPU 상태 유지.

    반환:
        tuple[AlignmentBatch, Optional[Tensor]]: 병합된 멀티-그래프 배치와
            결합된 정답 좌표 텐서(정답이 일부라도 없으면 None).

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 콜레이션.
    """
    if not items:
        raise ValueError("collate_alignment received no items.")

    batches = [it[0] for it in items]
    targets = [it[1] for it in items]

    query_pos = torch.cat([b.query_pos for b in batches], dim=0)
    query_atom_type = torch.cat([b.query_atom_type for b in batches], dim=0)
    query_batch = torch.cat(
        [torch.full((b.query_pos.size(0),), g, dtype=torch.long) for g, b in enumerate(batches)],
        dim=0,
    )

    # 결합 인덱스 재매핑을 위한 그래프별 원자 오프셋
    atom_counts = [b.query_pos.size(0) for b in batches]
    atom_offsets = [0]
    for n in atom_counts[:-1]:
        atom_offsets.append(atom_offsets[-1] + n)

    def cat_node(field: str) -> Optional[Tensor]:
        """모든 그래프에 존재하는 노드 필드를 dim=0으로 이어 붙인다.

        한 줄 요약:
            지정한 노드 필드를 모든 배치에서 모아, 전부 존재할 때만 연결한다.

        생성 이유:
            reference/pocket의 노드 속성처럼 일부 그래프에는 없을 수 있는 선택적
            노드 필드를 안전하게 합치기 위한 내부 헬퍼.

        역할:
            콜레이션 본문에서 선택적 노드 텐서 연결 로직의 중복을 제거한다.

        메커니즘:
            각 배치에서 ``field`` 속성을 꺼내 리스트로 모으고, ``_all_present``로
            전부 존재하면 ``torch.cat(dim=0)``으로 연결, 아니면 None을 반환한다.

        파라미터:
            field (str): 합칠 노드 필드(``AlignmentBatch`` 속성)의 이름.

        반환:
            Optional[Tensor]: 연결된 노드 텐서, 또는 일부 그래프에 없으면 None.

        파이프라인 단계:
            2단계(데이터 로딩/배치 구성)의 콜레이션 내부 보조.
        """
        vals = [getattr(b, field) for b in batches]
        return torch.cat(vals, dim=0) if _all_present(vals) else None

    def cat_with_batch(pos_field: str) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """좌표 필드를 이어 붙이고 대응하는 그래프 인덱스를 함께 생성한다.

        한 줄 요약:
            좌표 노드 필드를 모든 그래프에서 연결하고, 각 노드가 속한 그래프
            번호(batch 인덱스)를 동시에 만든다.

        생성 이유:
            reference/pocket 좌표는 합칠 때 어느 그래프에 속하는지 식별하는
            ``*_batch`` 인덱스가 반드시 함께 필요하므로, 좌표 연결과 인덱스
            생성을 한 번에 처리하는 헬퍼가 유용하다.

        역할:
            콜레이션 본문에서 ``(pos, batch_index)`` 쌍을 만드는 중복 로직을 묶는다.

        메커니즘:
            각 배치에서 ``pos_field`` 좌표를 모아 ``_all_present`` 검사 후, 전부
            존재하면 ``torch.cat(dim=0)``으로 좌표를 잇고 각 그래프 ``g``에
            ``torch.full``로 노드 수만큼 ``g``를 채운 인덱스를 dim=0으로 이어
            붙인다. 일부라도 없으면 ``(None, None)``을 반환한다.

        파라미터:
            pos_field (str): 합칠 좌표 필드(``AlignmentBatch`` 속성)의 이름.

        반환:
            tuple[Optional[Tensor], Optional[Tensor]]: ``(연결된 좌표, 그래프
                인덱스)``. 일부 그래프에 없으면 ``(None, None)``.

        파이프라인 단계:
            2단계(데이터 로딩/배치 구성)의 콜레이션 내부 보조.
        """
        vals = [getattr(b, pos_field) for b in batches]
        if not _all_present(vals):
            return None, None
        pos = torch.cat(vals, dim=0)
        idx = torch.cat(
            [torch.full((v.size(0),), g, dtype=torch.long) for g, v in enumerate(vals)],
            dim=0,
        )
        return pos, idx

    reference_pos, reference_batch = cat_with_batch("reference_pos")
    reference_atom_type = cat_node("reference_atom_type") if reference_pos is not None else None
    reference_node_attr = cat_node("reference_node_attr") if reference_pos is not None else None
    pocket_pos, pocket_batch = cat_with_batch("pocket_pos")
    pocket_atom_type = cat_node("pocket_atom_type") if pocket_pos is not None else None
    query_node_attr = cat_node("query_node_attr")

    # 결합: 각 그래프의 원자 인덱스에 오프셋을 적용한 후 이어 붙인다
    bond_indices = [b.query_bond_index for b in batches]
    bond_lengths = [b.query_bond_length for b in batches]
    if _all_present(bond_indices) and _all_present(bond_lengths):
        query_bond_index = torch.cat(
            [bi + atom_offsets[g] for g, bi in enumerate(bond_indices)], dim=1
        )
        query_bond_length = torch.cat(bond_lengths, dim=0)
    else:
        query_bond_index = None
        query_bond_length = None

    merged = AlignmentBatch(
        query_pos=query_pos,
        query_atom_type=query_atom_type,
        query_batch=query_batch,
        reference_pos=reference_pos,
        reference_atom_type=reference_atom_type,
        reference_batch=reference_batch,
        pocket_pos=pocket_pos,
        pocket_batch=pocket_batch,
        pocket_atom_type=pocket_atom_type,
        query_node_attr=query_node_attr,
        reference_node_attr=reference_node_attr,
        query_bond_index=query_bond_index,
        query_bond_length=query_bond_length,
    )
    merged = apply_conditioning(merged, conditioning)

    target = torch.cat(targets, dim=0) if _all_present(targets) else None

    if device is not None:
        merged = _batch_to_device(merged, device)
        if target is not None:
            target = target.to(device)
    return merged, target


def _batch_to_device(batch: AlignmentBatch, device: str | torch.device) -> AlignmentBatch:
    """배치의 모든 텐서 필드를 지정한 디바이스로 옮긴 새 배치를 만든다.

    한 줄 요약:
        ``AlignmentBatch``의 텐서 필드를 ``device``로 이동하고 비텐서 필드는
        그대로 둔 새 배치를 반환한다.

    생성 이유:
        콜레이션 결과를 GPU 등 학습/추론 디바이스로 옮겨야 하며, dataclass의
        텐서 필드만 선택적으로 이동하는 일관된 헬퍼가 필요하다.

    역할:
        파이프라인 2단계에서 배치를 계산 디바이스로 이동시킨다.

    메커니즘:
        ``dataclasses.fields``로 모든 필드를 순회하며, ``torch.is_tensor``인
        값만 ``value.to(device)``로 옮기고 나머지는 원본을 유지한 dict를 만들어
        ``AlignmentBatch(**moved)``로 재구성한다.

    파라미터:
        batch (AlignmentBatch): 디바이스를 옮길 입력 배치.
        device (str | torch.device): 목적지 디바이스.

    반환:
        AlignmentBatch: 텐서 필드가 ``device``로 이동된 새 배치.

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 디바이스 이동.
    """
    moved = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        moved[field.name] = value.to(device) if torch.is_tensor(value) else value
    return AlignmentBatch(**moved)


def sample_training_batch(
    dataset: AlignmentDataset,
    batch_size: int,
    conditioning: Conditioning,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[AlignmentBatch, Optional[Tensor]]:
    """``batch_size``개의 복합체를 비복원 추출로 무작위 선택하여 콜레이션한다.

    한 줄 요약:
        데이터셋에서 중복 없이 무작위로 ``batch_size``개 복합체를 뽑아 하나의
        학습 배치로 합친다.

    생성 이유:
        학습 루프가 매 스텝마다 무작위 미니배치를 필요로 하며, 데이터셋 크기보다
        큰 배치 요청도 안전하게 처리하는 샘플링 헬퍼가 필요하다.

    역할:
        파이프라인 2단계에서 데이터셋 -> 학습용 배치 샘플링과 콜레이션을 연결한다.

    메커니즘:
        ``k = min(batch_size, n)``로 요청 크기를 데이터셋 크기로 제한하고,
        ``torch.randperm(n, generator)``의 앞 ``k``개 인덱스로 비복원 추출한다.
        선택된 인덱스로 데이터셋에서 예제를 로드해 ``collate_alignment``로
        지정된 conditioning/디바이스에 맞춰 합친다.

    파라미터:
        dataset (AlignmentDataset): 샘플링 대상 데이터셋.
        batch_size (int): 뽑을 복합체 수(데이터셋 크기보다 크면 전부 사용).
        conditioning (Conditioning): 배치에 적용할 컨디셔닝 모드.
        device (str | torch.device | None): 결과를 이동할 디바이스(None이면 CPU).
        generator (torch.Generator | None): 추출 재현성을 위한 난수 생성기.

    반환:
        tuple[AlignmentBatch, Optional[Tensor]]: 콜레이션된 학습 배치와 정답
            좌표 텐서(없으면 None).

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 학습 배치 샘플링.
    """
    n = len(dataset)
    k = min(batch_size, n)
    idx = torch.randperm(n, generator=generator)[:k].tolist()
    items = [dataset[i] for i in idx]
    return collate_alignment(items, conditioning=conditioning, device=device)