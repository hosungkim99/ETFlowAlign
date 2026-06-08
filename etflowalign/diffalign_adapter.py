# etflowalign/diffalign_adapter.py
"""DiffAlign 스타일 분자 입력에서 ETFlowAlign .pt 페이로드로의 어댑터.

이 모듈은 의존성 경계를 단순하게 유지한다:
- DiffAlign은 mol_to_graph_data_obj를 통해 RDKit Mol -> PyG Data/Batch를 사용한다.
- ETFlowAlign은 etflowalign.data로 로드 가능한 순수 텐서 페이로드를 사용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data import Data

def load_rdkit_mol(path: str | Path, *, sanitize: bool = True, remove_hs: bool = True) -> Chem.Mol:
    """파일 확장자에 따라 SDF/MOL/PDB에서 RDKit 분자를 로드한다.

    생성이유:
        외부 diffalign 예제 입력(리간드/포켓 파일)을 ETFlowAlign 페이로드로
        변환하려면 먼저 다양한 포맷의 분자 파일을 컨포머 좌표와 함께 안전하게
        읽어들이는 통일된 로더가 필요하다.

    역할:
        보조(11단계) 외부 연동의 입력 로딩 책임. 확장자별로 알맞은 RDKit 리더를
        선택하고, 로드 실패 및 좌표 부재를 명시적으로 검증한다.

    메커니즘:
        경로 확장자를 소문자로 보고 .sdf/.mol이면 SDMolSupplier의 첫 분자를,
        .pdb면 MolFromPDBFile을 사용한다. 지원하지 않는 확장자는 ValueError.
        로드 결과가 None이면 ValueError, 컨포머가 0개면 좌표 부재 ValueError.

    파라미터:
        path (str | Path): 분자 파일 경로(.sdf/.mol/.pdb).
        sanitize (bool): RDKit sanitize 수행 여부(키워드 전용). 기본 True.
        remove_hs (bool): 수소 원자 제거 여부(키워드 전용). 기본 True.

    반환:
        Chem.Mol: 컨포머 좌표를 가진 RDKit 분자 객체.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 입력 분자 로딩.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
        mol = supplier[0] if supplier and len(supplier) > 0 else None
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), sanitize=sanitize, removeHs=remove_hs)
    else:
        raise ValueError(f"Unsupported molecule file extension: {path.suffix}")

    if mol is None:
        raise ValueError(f"Failed to load molecule: {path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule has no conformer coordinates: {path}")

    return mol


def rdkit_mol_to_pos(mol: Chem.Mol) -> Tensor:
    """RDKit 컨포머 좌표를 Tensor[N,3]으로 변환한다.

    생성이유:
        ETFlowAlign는 순수 텐서 페이로드로 동작하므로 RDKit 컨포머의 3D 좌표를
        torch 텐서로 옮기는 변환기가 필요하다.

    역할:
        보조(11단계)에서 분자 좌표를 텐서로 추출한다(예: 포켓 좌표 추출).

    메커니즘:
        분자의 첫 컨포머를 얻고, 원자 인덱스 순으로 GetAtomPosition의 x,y,z를
        리스트에 모은 뒤 float32 텐서로 변환한다.

    파라미터:
        mol (Chem.Mol): 컨포머를 가진 RDKit 분자.

    반환:
        Tensor: 원자 좌표 ``[N, 3]`` (float32).

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 좌표 텐서화.
    """
    conf = mol.GetConformer()
    coords = []

    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        coords.append([pos.x, pos.y, pos.z])

    return torch.tensor(coords, dtype=torch.float32)

def mol_to_node_attr_local(mol: Chem.Mol) -> Tensor:
    """ETFlowAlign 노드 컨디셔닝을 위한 최소한의 RDKit 원자 특성을 구성한다.

    특성 순서:
        [atomic_number, atom_degree, formal_charge, is_aromatic, is_in_ring]

    생성이유:
        모델의 노드 컨디셔닝 입력으로 쓸, 분자에 독립적인 5차원 화학 특성
        벡터가 필요하다. diffalign 예제와 동일한 특성 정의를 로컬에 둔다.

    역할:
        보조(11단계)에서 각 원자에 대한 [N, 5] 노드 속성(node_attr)을 만든다.

    메커니즘:
        분자의 모든 원자를 순회하며 원자번호, 차수, 형식전하, 방향족 여부(0/1),
        고리 포함 여부(0/1)를 float로 모아 float32 텐서 ``[N, 5]``로 만든다.

    파라미터:
        mol (Chem.Mol): 원자 특성을 추출할 RDKit 분자.

    반환:
        Tensor: 노드 특성 행렬 ``[N, 5]`` (float32).

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 노드 특성 구성.
    """
    features = []
    for atom in mol.GetAtoms():
        features.append(
            [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
            ]
        )

    return torch.tensor(features, dtype=torch.float32)

def _bond_type_to_int(bond: Chem.Bond) -> int:
    """RDKit 결합 타입을 정수 코드로 매핑한다.

    생성이유:
        그래프 텐서의 edge_type 필드는 정수/실수 코드여야 하므로 RDKit의
        BondType 열거형을 정해진 정수 규약으로 변환해야 한다.

    역할:
        보조(11단계)에서 결합 타입을 그래프 엣지 속성으로 인코딩한다.

    메커니즘:
        SINGLE→1, DOUBLE→2, TRIPLE→3, AROMATIC→12로 매핑한다. 그 외 타입은
        지원하지 않으므로 ValueError를 발생시킨다.

    파라미터:
        bond (Chem.Bond): 변환할 RDKit 결합 객체.

    반환:
        int: 결합 타입 정수 코드(1/2/3/12).

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 결합 타입 인코딩 보조.
    """
    bond_type = bond.GetBondType()

    if bond_type == Chem.rdchem.BondType.SINGLE:
        return 1
    if bond_type == Chem.rdchem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.rdchem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.rdchem.BondType.AROMATIC:
        return 12

    raise ValueError(f"Unsupported bond type: {bond_type}")

def mol_to_graph_data_obj_local(mol: Chem.Mol) -> Data:
    """DiffAlign의 RDKit Mol -> PyG Data 변환의 로컬 복사본.

    출력 필드:
        atom_type: Tensor[N], 원자 번호
        edge_index: Tensor[2, 2E], 방향 있는 결합 엣지
        edge_type: Tensor[2E], float 형태의 결합 타입
        pos: Tensor[N, 3], 컨포머 좌표

    생성이유:
        외부 diffalign 패키지에 직접 의존하지 않으면서 동일한 RDKit→PyG Data
        변환 결과를 얻기 위해, 해당 변환을 로컬에 복제해 의존성 경계를 단순화한다.

    역할:
        보조(11단계)에서 RDKit 분자를 PyG Data(노드/엣지/좌표/노드속성)로
        변환한다. 이후 Batch.from_data_list로 묶여 페이로드 변환에 사용된다.

    메커니즘:
        원자번호로 atom_type(long)을 만들고, 각 결합을 양방향 엣지 2개로 펼쳐
        edge_index(2×2E)와 edge_type(float)을 구성한다. 결합이 없으면 빈 텐서.
        첫 컨포머에서 좌표 pos(N×3)를 모으고, mol_to_node_attr_local로 node_attr
        (N×5)을 만든 뒤 이들을 묶은 Data 객체를 반환한다.

    파라미터:
        mol (Chem.Mol): 변환할 RDKit 분자(컨포머 보유).

    반환:
        Data: atom_type/edge_index/edge_type/pos/node_attr를 가진 PyG Data.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 그래프 데이터 변환.
    """
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    atom_type = torch.tensor(atom_features, dtype=torch.long)

    edges = []
    bond_types = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = _bond_type_to_int(bond)

        edges.append((i, j))
        bond_types.append(bond_type)
        edges.append((j, i))
        bond_types.append(bond_type)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(bond_types, dtype=torch.float32).view(-1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.float32)

    conf = mol.GetConformer()
    coordinates = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coordinates.append([pos.x, pos.y, pos.z])

    pos = torch.tensor(coordinates, dtype=torch.float32)
    node_attr = mol_to_node_attr_local(mol)

    return Data(
        atom_type=atom_type,
        edge_index=edge_index,
        edge_type=edge_type,
        pos=pos,
        node_attr=node_attr,
    )
    
def mol_to_bond_index_and_length_local(mol: Chem.Mol, pos: Tensor) -> tuple[Tensor, Tensor]:
    """비방향 쿼리 결합 인덱스와 목표 결합 길이를 구성한다.

    Returns:
        bond_index: LongTensor[2, E], RDKit 결합당 하나의 비방향 엣지.
        bond_length: FloatTensor[E], 각 결합의 컨포머 결합 길이.

    생성이유:
        학습 목표에서 리간드 결합 기하(결합 길이)를 보존/감독하려면, 각 결합에
        대한 비방향 인덱스와 기준 결합 길이가 필요하다.

    역할:
        보조(11단계)에서 (학습) 페이로드에 들어갈 query_bond_index/
        query_bond_length를 만든다.

    메커니즘:
        각 결합의 시작/끝 원자 인덱스 쌍을 모아 비방향 엣지로 두고, 주어진
        좌표 pos에서 두 원자 간 노름(거리)을 결합 길이로 계산한다. 결합이
        없으면 빈 텐서 쌍을 반환한다. 있으면 인덱스를 2×E long 텐서로,
        길이를 pos dtype의 E 벡터로 만든다.

    파라미터:
        mol (Chem.Mol): 결합 목록을 제공하는 RDKit 분자.
        pos (Tensor): 결합 길이를 계산할 기준 좌표 ``[N, 3]``.

    반환:
        tuple[Tensor, Tensor]: (bond_index ``[2, E]`` long,
            bond_length ``[E]`` float).

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 결합 기하 구성.
    """
    edges = []
    lengths = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append((i, j))
        lengths.append(torch.linalg.norm(pos[i] - pos[j]))

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=pos.dtype),
        )

    bond_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    bond_length = torch.stack(lengths).to(dtype=pos.dtype)

    return bond_index, bond_length    
    
def _make_cpu_generator(seed: int | None) -> torch.Generator | None:
    """주어진 시드로 CPU 난수 생성기를 만들어 재현성을 확보한다.

    생성이유:
        쿼리 좌표 랜덤 강체 변환 등의 난수 연산을 시드로 재현 가능하게 만들기
        위해 명시적 Generator가 필요하다.

    역할:
        보조(11단계)에서 랜덤 회전/이동에 사용할 결정론적 난수원을 제공한다.

    메커니즘:
        seed가 None이면 None을 반환(전역 RNG 사용). 아니면 CPU Generator를
        만들고 manual_seed로 시드를 고정해 반환한다.

    파라미터:
        seed (int | None): 난수 시드. None이면 생성기를 만들지 않는다.

    반환:
        torch.Generator | None: 시드된 CPU 생성기 또는 None.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 난수 재현성 보조.
    """
    if seed is None:
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator

def _random_rotation_matrix(
    *,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample a proper 3D rotation matrix on CPU.

    한 줄 요약:
        CPU에서 단위 사원수를 샘플링해 유효한 3D 회전 행렬(SO(3))을 만든다.

    생성이유:
        소스 쿼리 좌표를 결합 기하를 깨지 않으면서 무작위로 배향시키려면,
        반사가 섞이지 않은 적법한 회전 행렬이 필요하다.

    역할:
        보조(11단계)에서 _make_query_source_pos가 사용하는 랜덤 회전을 생성한다.

    메커니즘:
        4차원 정규분포 샘플 q를 정규화해 단위 사원수를 만들고(w,x,y,z),
        사원수→회전행렬 공식을 적용해 3×3 회전 행렬을 구성한다.

    파라미터:
        dtype (torch.dtype): 출력 행렬의 자료형(키워드 전용).
        generator (torch.Generator | None): 재현성을 위한 난수 생성기(키워드 전용).

    반환:
        Tensor: SO(3) 회전 행렬 ``[3, 3]``.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 랜덤 회전 보조.
    """
    q = torch.randn(4, dtype=dtype, generator=generator)
    q = q / q.norm().clamp_min(1e-8)

    w, x, y, z = q.unbind()

    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ],
        dim=0,
    )


def _make_query_source_pos(
    centered_query_pos: Tensor,
    *,
    randomize_query_pos: bool,
    seed: int | None,
    translation_scale: float = 1.0,
) -> Tensor:
    """Build source query coordinates without breaking ligand bond geometry.

    If randomize_query_pos=True, apply a random rigid transform to the centered
    query conformer. If False, return the centered query conformer itself.

    한 줄 요약:
        리간드 결합 기하를 보존하면서 flow matching의 소스 쿼리 좌표를 만든다.

    생성이유:
        샘플링/학습의 출발점이 되는 소스 좌표는 결합 길이/각을 깨지 않아야
        한다. 강체 변환(회전+이동)만 적용해 기하를 보존하면서 다양성을 준다.

    역할:
        보조(11단계)에서 추론/학습 페이로드의 query_pos(소스 좌표)를 생성한다.

    메커니즘:
        randomize_query_pos가 False면 중심 정렬된 입력 좌표를 그대로 clone한다.
        True면 시드된 생성기로 랜덤 회전 행렬과 랜덤 이동 벡터(translation_scale
        배)를 만들어, 좌표를 자기 중심으로 옮긴 뒤 회전(@R.T)하고 중심을
        되돌린 다음 이동을 더한다(강체 변환).

    파라미터:
        centered_query_pos (Tensor): 중심 정렬된 쿼리 컨포머 좌표 ``[N, 3]``.
        randomize_query_pos (bool): 랜덤 강체 변환 적용 여부(키워드 전용).
        seed (int | None): 난수 시드(키워드 전용).
        translation_scale (float): 랜덤 이동 벡터의 스케일. 기본 1.0.

    반환:
        Tensor: 소스 쿼리 좌표 ``[N, 3]``.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 소스 좌표 생성.
    """
    if not randomize_query_pos:
        return centered_query_pos.clone()

    generator = _make_cpu_generator(seed)
    rotation = _random_rotation_matrix(
        dtype=centered_query_pos.dtype,
        generator=generator,
    )

    center = centered_query_pos.mean(dim=0, keepdim=True)
    translation = torch.randn(
        1,
        3,
        dtype=centered_query_pos.dtype,
        generator=generator,
    ) * float(translation_scale)

    return (centered_query_pos - center) @ rotation.T + center + translation

def shift_rdkit_mol_inplace(mol: Chem.Mol, shift: Tensor) -> Chem.Mol:
    """Shift RDKit conformer coordinates in-place by subtracting shift.

    한 줄 요약:
        RDKit 컨포머 좌표를 shift만큼 빼서 제자리(in-place)로 이동시킨다.

    생성이유:
        포켓 좌표를 reference 평균 중심 좌표계로 맞추려면 동일한 중심값을 빼야
        한다. 이를 RDKit 분자에 직접 적용하는 유틸이 필요하다.

    역할:
        보조(11단계)에서 포켓 분자를 reference 중심 좌표계로 정렬한다.

    메커니즘:
        shift를 CPU float로 변환해 Point3D로 만들고, 분자의 첫 컨포머의 모든
        원자 위치에서 이 벡터를 빼서 SetAtomPosition으로 갱신한다. 변경된 동일
        분자 객체를 반환한다.

    파라미터:
        mol (Chem.Mol): 좌표를 이동시킬 RDKit 분자(제자리 변경됨).
        shift (Tensor): 빼낼 이동 벡터 ``[3]``.

    반환:
        Chem.Mol: 좌표가 이동된 동일 분자 객체.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 포켓 좌표 정렬.
    """
    shift = shift.detach().cpu().float()
    shift_vec = Point3D(float(shift[0]), float(shift[1]), float(shift[2]))

    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        old = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, old - shift_vec)

    return mol


def diffalign_batches_to_etflowalign_payload(
    query_batch: Batch,
    reference_batch: Batch,
    *,
    pocket_mol: Chem.Mol | None = None,
    target_query_pos: Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert DiffAlign PyG Batch objects into an ETFlowAlign tensor payload.

    한 줄 요약:
        DiffAlign PyG Batch(쿼리/참조)를 ETFlowAlign가 로드 가능한 텐서
        페이로드(dict)로 변환한다.

    생성이유:
        diffalign의 그래프 표현과 ETFlowAlign의 순수 텐서 페이로드 사이를 잇는
        단일 변환 지점이 필요하다. 필수 필드 검증과 dtype/디바이스 정규화를 한다.

    역할:
        보조(11단계) 외부 연동의 핵심 변환 함수. 쿼리/참조 좌표·원자타입·배치
        인덱스를 추출하고, 선택적으로 노드속성/학습타깃/포켓/메타데이터를 덧붙인다.

    메커니즘:
        query_batch와 reference_batch에 pos/atom_type/batch 필드가 있는지 검사하고
        없으면 ValueError. 각 텐서를 detach·cpu·적절한 dtype(float/long)으로 옮겨
        payload dict에 담는다. node_attr가 있으면 query/reference_node_attr 추가.
        target_query_pos가 주어지면 query_pos와 형상이 같은지 검증 후 추가.
        pocket_mol이 있으면 좌표를 추출해 pocket_pos와 0으로 채운 pocket_batch
        추가. metadata가 있으면 최상위 키로 병합한다.

    반환(파이프라인 요약):
        torch.save로 저장하고 etflowalign.data.load_alignment_batch_from_pt로
        로드 가능한 dict 페이로드.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 배치→페이로드 변환.

    Args:
        query_batch:
            DiffAlign query ligand PyG Batch.
        reference_batch:
            DiffAlign reference ligand PyG Batch.
        pocket_mol:
            Optional RDKit pocket molecule. Coordinates should already be in the
            same coordinate frame as query/reference.
        target_query_pos:
            Optional target query coordinates for training. Inference payloads
            should omit this field.
        metadata:
            Optional extra metadata stored as top-level keys in the .pt payload.

    Returns:
        Dict that can be saved by torch.save and loaded with
        etflowalign.data.load_alignment_batch_from_pt.
    """
    required_query_fields = ["pos", "atom_type", "batch"]
    required_reference_fields = ["pos", "atom_type", "batch"]

    for name in required_query_fields:
        if not hasattr(query_batch, name):
            raise ValueError(f"query_batch is missing required field '{name}'")

    for name in required_reference_fields:
        if not hasattr(reference_batch, name):
            raise ValueError(f"reference_batch is missing required field '{name}'")

    payload: dict[str, Any] = {
        "query_pos": query_batch.pos.detach().cpu().float(),
        "query_atom_type": query_batch.atom_type.detach().cpu().long(),
        "query_batch": query_batch.batch.detach().cpu().long(),
        "reference_pos": reference_batch.pos.detach().cpu().float(),
        "reference_atom_type": reference_batch.atom_type.detach().cpu().long(),
        "reference_batch": reference_batch.batch.detach().cpu().long(),
    }
    if hasattr(query_batch, "node_attr"):
        payload["query_node_attr"] = query_batch.node_attr.detach().cpu().float()

    if hasattr(reference_batch, "node_attr"):
        payload["reference_node_attr"] = reference_batch.node_attr.detach().cpu().float()
        
    if target_query_pos is not None:
        if target_query_pos.shape != payload["query_pos"].shape:
            raise ValueError(
                "target_query_pos must match query_pos shape. "
                f"target={tuple(target_query_pos.shape)}, query={tuple(payload['query_pos'].shape)}"
            )
        payload["target_query_pos"] = target_query_pos.detach().cpu().float()

    if pocket_mol is not None:
        pocket_pos = rdkit_mol_to_pos(pocket_mol)
        payload["pocket_pos"] = pocket_pos
        payload["pocket_batch"] = torch.zeros(pocket_pos.size(0), dtype=torch.long)

    if metadata:
        payload.update(metadata)

    return payload


def build_diffalign_example_inference_payload(
    *,
    repo_root: str | Path,
    query_sdf: str | Path,
    reference_sdf: str | Path,
    pocket_pdb: str | Path | None = None,
    randomize_query_pos: bool = True,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Build an ETFlowAlign inference payload from DiffAlign example files.

    한 줄 요약:
        DiffAlign 예제 파일(쿼리/참조 SDF, 선택적 포켓 PDB)로부터 ETFlowAlign
        추론용 텐서 페이로드를 만든다.

    생성이유:
        실제 분자 파일에서 곧바로 추론 입력을 만들 수 있어야 빠르게 엔드투엔드
        파이프라인을 검증할 수 있다. 좌표계 정렬 규약을 일관되게 적용한다.

    역할:
        보조(11단계)의 추론 페이로드 빌더. 파일 로드→그래프 변환→reference
        평균 중심화→소스 쿼리 생성→페이로드 변환을 하나로 묶는다(학습 타깃 없음).

    메커니즘:
        쿼리/참조(필요시 포켓) 분자를 로드하고 그래프 Data로 변환한다.
        reference 평균(reference_mean)으로 reference 좌표와 쿼리 좌표를 중심화
        하고, 포켓이 있으면 shift_rdkit_mol_inplace로 같은 중심을 적용한다.
        Batch.from_data_list로 묶은 뒤 _make_query_source_pos로 (선택적 랜덤
        강체 변환된) 소스 query_pos를 만들고,
        diffalign_batches_to_etflowalign_payload에 target 없이 넘겨 메타데이터와
        함께 페이로드를 생성한다.

    파라미터:
        repo_root (str | Path): 저장소 루트 경로(절대경로로 resolve, 키워드 전용).
        query_sdf (str | Path): 쿼리 리간드 SDF 경로(키워드 전용).
        reference_sdf (str | Path): 참조 리간드 SDF 경로(키워드 전용).
        pocket_pdb (str | Path | None): 포켓 PDB 경로 또는 None(키워드 전용).
        randomize_query_pos (bool): 소스 쿼리에 랜덤 강체 변환 적용 여부. 기본 True.
        seed (int | None): 난수 시드. 기본 0.

    반환:
        dict[str, Any]: 추론용 ETFlowAlign 페이로드.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 추론 페이로드 빌드.

    Coordinate convention:
    - reference coordinates are centered by reference mean.
    - pocket coordinates are shifted by the same reference mean.
    - query coordinates are in the same centered frame.
    - randomize_query_pos=True applies a rigid random transform to the centered
      query conformer, preserving bond geometry.
    """
    repo_root = Path(repo_root).resolve()

    query_mol = load_rdkit_mol(query_sdf, sanitize=True, remove_hs=True)
    reference_mol = load_rdkit_mol(reference_sdf, sanitize=True, remove_hs=True)

    pocket_mol = None
    if pocket_pdb is not None:
        pocket_mol = load_rdkit_mol(pocket_pdb, sanitize=True, remove_hs=True)

    query_data = mol_to_graph_data_obj_local(query_mol)
    reference_data = mol_to_graph_data_obj_local(reference_mol)

    reference_mean = reference_data.pos.mean(dim=0)
    centered_query_pos = query_data.pos.clone() - reference_mean
    reference_data.pos = reference_data.pos - reference_mean

    if pocket_mol is not None:
        pocket_mol = Chem.Mol(pocket_mol)
        shift_rdkit_mol_inplace(pocket_mol, reference_mean)

    query_batch = Batch.from_data_list([query_data])
    reference_batch = Batch.from_data_list([reference_data])
    query_batch.pos = _make_query_source_pos(
        centered_query_pos,
        randomize_query_pos=randomize_query_pos,
        seed=seed,
    )

    payload = diffalign_batches_to_etflowalign_payload(
        query_batch=query_batch,
        reference_batch=reference_batch,
        pocket_mol=pocket_mol,
        target_query_pos=None,
        metadata={
            "source": "diffalign_example",
            "query_sdf": str(query_sdf),
            "reference_sdf": str(reference_sdf),
            "pocket_pdb": str(pocket_pdb) if pocket_pdb is not None else None,
            "reference_center_subtracted": reference_mean.detach().cpu(),
            "randomize_query_pos": bool(randomize_query_pos),
            "seed": seed,
            "query_source": "rigid_randomized_query_conformer"if randomize_query_pos 
            else "centered_query_conformer",
        },
    )

    return payload

def build_diffalign_example_train_payload(
    *,
    repo_root: str | Path,
    query_sdf: str | Path,
    reference_sdf: str | Path,
    pocket_pdb: str | Path | None = None,
    randomize_query_pos: bool = True,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Build an ETFlowAlign pseudo-training payload from DiffAlign example files.

    한 줄 요약:
        DiffAlign 예제 파일로부터 학습 타깃(target_query_pos)과 결합 기하를
        포함한 ETFlowAlign 의사-학습 페이로드를 만든다.

    생성이유:
        엔드투엔드 학습 경로를 검증하려면 소스 좌표뿐 아니라 정답 타깃과 결합
        길이까지 갖춘 페이로드가 필요하다.

    역할:
        보조(11단계)의 학습 페이로드 빌더. 추론 빌더와 유사하나 타깃과
        query_bond_index/query_bond_length를 추가로 구성한다.

    메커니즘:
        분자를 로드/변환하고 reference_mean을 구한다. target_query_pos =
        쿼리 컨포머 - reference_mean(쿼리 형상 [N_query,3])을 만든 뒤,
        mol_to_bond_index_and_length_local로 결합 인덱스/길이를 계산한다.
        reference(및 포켓) 좌표를 같은 중심 좌표계로 옮기고, Batch로 묶은 후
        _make_query_source_pos(target 기준, 선택적 랜덤 강체 변환)로 소스
        query_pos를 만든다. diffalign_batches_to_etflowalign_payload에 target과
        메타데이터를 넘겨 페이로드를 만들고, query_bond_index/query_bond_length를
        cpu long/float로 덧붙여 반환한다.

    파라미터:
        repo_root (str | Path): 저장소 루트 경로(키워드 전용).
        query_sdf (str | Path): 쿼리 리간드 SDF 경로(키워드 전용).
        reference_sdf (str | Path): 참조 리간드 SDF 경로(키워드 전용).
        pocket_pdb (str | Path | None): 포켓 PDB 경로 또는 None(키워드 전용).
        randomize_query_pos (bool): 소스 쿼리에 랜덤 강체 변환 적용 여부. 기본 True.
        seed (int | None): 난수 시드. 기본 0.

    반환:
        dict[str, Any]: 타깃과 결합 기하를 포함한 학습용 ETFlowAlign 페이로드.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 학습 페이로드 빌드.

    Training target convention:
        target_query_pos = original_query_conformer - reference_center

    Source convention:
        query_pos = random rigid transform of target_query_pos
        if randomize_query_pos=True;
        otherwise query_pos = target_query_pos.
    """
    repo_root = Path(repo_root).resolve()

    query_mol = load_rdkit_mol(query_sdf, sanitize=True, remove_hs=True)
    reference_mol = load_rdkit_mol(reference_sdf, sanitize=True, remove_hs=True)

    pocket_mol = None
    if pocket_pdb is not None:
        pocket_mol = load_rdkit_mol(pocket_pdb, sanitize=True, remove_hs=True)

    query_data = mol_to_graph_data_obj_local(query_mol)
    reference_data = mol_to_graph_data_obj_local(reference_mol)

    reference_mean = reference_data.pos.mean(dim=0)

    # Important: target must be query-shaped [N_query, 3], not reference-shaped.
    target_query_pos = query_data.pos.clone() - reference_mean
    query_bond_index, query_bond_length = mol_to_bond_index_and_length_local(
    query_mol,
    target_query_pos,
    )
    # Put reference and pocket into the same centered coordinate frame.
    reference_data.pos = reference_data.pos - reference_mean

    if pocket_mol is not None:
        pocket_mol = Chem.Mol(pocket_mol)
        shift_rdkit_mol_inplace(pocket_mol, reference_mean)

    query_batch = Batch.from_data_list([query_data])
    reference_batch = Batch.from_data_list([reference_data])

    query_batch.pos = _make_query_source_pos(
        target_query_pos,
        randomize_query_pos=randomize_query_pos,
        seed=seed,
    )

    payload = diffalign_batches_to_etflowalign_payload(
        query_batch=query_batch,
        reference_batch=reference_batch,
        pocket_mol=pocket_mol,
        target_query_pos=target_query_pos,
        metadata={
            "source": "diffalign_example_pseudo_train",
            "query_sdf": str(query_sdf),
            "reference_sdf": str(reference_sdf),
            "pocket_pdb": str(pocket_pdb) if pocket_pdb is not None else None,
            "reference_center_subtracted": reference_mean.detach().cpu(),
            "target_definition": "query_sdf_conformer_minus_reference_center",
            "randomize_query_pos": bool(randomize_query_pos),
            "query_source": "rigid_randomized_query_conformer" if randomize_query_pos 
            else "centered_query_conformer",
            "seed": seed,
        },
    )
    
    payload["query_bond_index"] = query_bond_index.detach().cpu().long()
    payload["query_bond_length"] = query_bond_length.detach().cpu().float()
    
    return payload

def _require_tensor(payload: dict[str, object], key: str) -> torch.Tensor:
    """Return a required tensor field from a payload.

    This helper keeps CLI scripts short while making payload errors explicit.

    한 줄 요약:
        페이로드에서 필수 텐서 필드를 꺼내 존재/타입을 검증해 반환한다.

    생성이유:
        페이로드 검증 코드를 짧게 유지하면서 누락/타입 오류를 명확한 예외로
        드러내기 위한 공통 헬퍼가 필요하다.

    역할:
        보조(11단계) 페이로드 검증 함수들의 기반 추출기.

    메커니즘:
        key가 페이로드에 없으면 KeyError. 값이 torch 텐서가 아니면 TypeError.
        둘 다 통과하면 해당 텐서를 반환한다.

    파라미터:
        payload (dict[str, object]): 검사할 페이로드 딕셔너리.
        key (str): 가져올 필수 필드 이름.

    반환:
        torch.Tensor: 검증된 텐서 값.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 페이로드 검증 보조.
    """
    if key not in payload:
        raise KeyError(f"Missing required payload key: {key}")

    value = payload[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")

    return value

def _validate_pos_tensor(name: str, pos: torch.Tensor) -> None:
    """Validate coordinate tensor shape [N, 3].

    한 줄 요약:
        좌표 텐서가 ``[N, 3]`` 형상이고 부동소수 dtype인지 검증한다.

    생성이유:
        좌표 페이로드의 형상/타입 오류를 로딩 전에 명확히 차단하기 위함이다.

    역할:
        보조(11단계)에서 query/reference/pocket/target 좌표 검증에 재사용된다.

    메커니즘:
        2차원이 아니거나 마지막 차원이 3이 아니면 ValueError. dtype이
        float16/32/64가 아니면 TypeError.

    파라미터:
        name (str): 오류 메시지에 쓸 텐서 이름.
        pos (torch.Tensor): 검증할 좌표 텐서.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 좌표 형상 검증.
    """
    if pos.dim() != 2 or pos.size(1) != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {tuple(pos.shape)}")
    if pos.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"{name} must be a floating tensor, got {pos.dtype}")

def _validate_batch_tensor(name: str, batch: torch.Tensor, pos: torch.Tensor) -> None:
    """Validate batch index tensor shape [N].

    한 줄 요약:
        배치 인덱스 텐서가 ``[N]`` 형상, 좌표와 같은 N, long dtype인지 검증한다.

    생성이유:
        그래프 분할 인덱스의 형상/길이/타입 불일치는 후속 모델 연산을 깨뜨리므로
        사전 검증이 필요하다.

    역할:
        보조(11단계)에서 query/reference/pocket batch 텐서 검증에 재사용된다.

    메커니즘:
        1차원이 아니면 ValueError. 길이가 pos의 첫 차원과 다르면 ValueError.
        dtype이 long이 아니면 TypeError.

    파라미터:
        name (str): 오류 메시지에 쓸 텐서 이름.
        batch (torch.Tensor): 검증할 배치 인덱스 텐서.
        pos (torch.Tensor): 길이 비교 기준이 되는 좌표 텐서.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 배치 인덱스 검증.
    """
    if batch.dim() != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(batch.shape)}")
    if batch.size(0) != pos.size(0):
        raise ValueError(
            f"{name} and position tensor must have the same first dimension: "
            f"{tuple(batch.shape)} vs {tuple(pos.shape)}"
        )
    if batch.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {batch.dtype}")

def _validate_atom_type_tensor(name: str, atom_type: torch.Tensor, pos: torch.Tensor) -> None:
    """Validate atom type tensor shape [N].

    한 줄 요약:
        원자 타입 텐서가 ``[N]`` 형상, 좌표와 같은 N, long dtype인지 검증한다.

    생성이유:
        원자 타입 길이가 좌표와 어긋나면 노드 컨디셔닝이 잘못되므로 사전 검증한다.

    역할:
        보조(11단계)에서 query/reference atom_type 검증에 재사용된다.

    메커니즘:
        1차원이 아니면 ValueError. 길이가 pos 첫 차원과 다르면 ValueError.
        dtype이 long이 아니면 TypeError.

    파라미터:
        name (str): 오류 메시지에 쓸 텐서 이름.
        atom_type (torch.Tensor): 검증할 원자 타입 텐서.
        pos (torch.Tensor): 길이 비교 기준이 되는 좌표 텐서.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 원자 타입 검증.
    """
    if atom_type.dim() != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(atom_type.shape)}")
    if atom_type.size(0) != pos.size(0):
        raise ValueError(
            f"{name} and position tensor must have the same first dimension: "
            f"{tuple(atom_type.shape)} vs {tuple(pos.shape)}"
        )
    if atom_type.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {atom_type.dtype}")

def _validate_node_attr_tensor(name: str, node_attr: torch.Tensor, pos: torch.Tensor) -> None:
    """Validate node attribute tensor shape [N, 5].

    한 줄 요약:
        노드 속성 텐서가 ``[N, 5]`` 형상, 좌표와 같은 N, float32 dtype인지 검증한다.

    생성이유:
        노드 컨디셔닝 특성은 정확히 5차원(원자번호/차수/형식전하/방향족/고리)
        이어야 하므로 형상/타입을 강하게 검증한다.

    역할:
        보조(11단계)에서 query/reference node_attr 검증에 재사용된다.

    메커니즘:
        2차원이 아니면 ValueError. 행 수가 pos와 다르면 ValueError. 특성 차원이
        5가 아니면 ValueError. dtype이 float32가 아니면 TypeError.

    파라미터:
        name (str): 오류 메시지에 쓸 텐서 이름.
        node_attr (torch.Tensor): 검증할 노드 속성 텐서.
        pos (torch.Tensor): 행 수 비교 기준이 되는 좌표 텐서.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 노드 속성 검증.
    """
    if node_attr.dim() != 2:
        raise ValueError(f"{name} must have shape [N, F], got {tuple(node_attr.shape)}")
    if node_attr.size(0) != pos.size(0):
        raise ValueError(
            f"{name} and position tensor must have the same first dimension: "
            f"{tuple(node_attr.shape)} vs {tuple(pos.shape)}"
        )
    if node_attr.size(1) != 5:
        raise ValueError(f"{name} feature dimension must be 5, got {node_attr.size(1)}")
    if node_attr.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {node_attr.dtype}")

def _validate_reference_center_metadata(payload: dict[str, object]) -> None:
    """Validate reference_center_subtracted metadata.

    한 줄 요약:
        페이로드에 reference_center_subtracted 메타데이터가 ``[3]`` 텐서로
        기록되어 있는지 검증한다.

    생성이유:
        중심화에 사용한 reference 중심값을 보존해야 추론 결과를 원래 좌표계로
        되돌리거나 추적할 수 있다. 그 기록 여부/형식을 강제한다.

    역할:
        보조(11단계)에서 좌표계 정렬 메타데이터의 존재와 형식을 검증한다.

    메커니즘:
        reference_center_subtracted 키가 없으면 KeyError. 값이 텐서가 아니면
        TypeError. 형상이 ``(3,)``이 아니면 ValueError.

    파라미터:
        payload (dict[str, object]): 검사할 페이로드 딕셔너리.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 중심 메타데이터 검증.
    """
    if "reference_center_subtracted" not in payload:
        raise KeyError(
            "Missing metadata key: reference_center_subtracted. "
            "DiffAlign example payloads should record the reference center used for centering."
        )

    center = payload["reference_center_subtracted"]
    if not torch.is_tensor(center):
        raise TypeError(
            "reference_center_subtracted must be a torch.Tensor, "
            f"got {type(center)}"
        )
    if center.shape != (3,):
        raise ValueError(
            "reference_center_subtracted must have shape [3], "
            f"got {tuple(center.shape)}"
        )


def validate_diffalign_example_inference_payload(payload: dict[str, object]) -> None:
    """Validate a DiffAlign-example inference payload.

    한 줄 요약:
        DiffAlign 예제 추론 페이로드의 모든 필수 필드와 형상/타입/규약을 검증한다.

    생성이유:
        잘못된 페이로드가 추론에 들어가면 디버깅이 어려운 실패를 낳는다.
        로딩 전 단일 검증 진입점에서 모든 규약을 강제한다.

    역할:
        보조(11단계)에서 추론 페이로드의 정합성을 종합 검증한다.

    메커니즘:
        _require_tensor로 query/reference/pocket의 pos·atom_type·batch·node_attr를
        가져온다. _validate_pos_tensor로 세 좌표를, _validate_atom_type_tensor로
        원자 타입을, _validate_batch_tensor로 배치 인덱스를,
        _validate_node_attr_tensor로 노드 속성을 검증한다. pocket_pos가 비어
        있으면 ValueError(실제 포켓 좌표 요구). 마지막으로
        _validate_reference_center_metadata로 중심 메타데이터를 검증한다.

    파라미터:
        payload (dict[str, object]): 검증할 추론 페이로드.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 추론 페이로드 종합 검증.

    Required conventions:
    - query/reference coordinates use the same centered coordinate frame.
    - pocket coordinates are real coordinates from pocket.pdb, not an empty placeholder.
    - query_node_attr/reference_node_attr are RDKit chemistry features with shape [N, 5].
    """

    query_pos = _require_tensor(payload, "query_pos")
    query_atom_type = _require_tensor(payload, "query_atom_type")
    query_batch = _require_tensor(payload, "query_batch")
    query_node_attr = _require_tensor(payload, "query_node_attr")

    reference_pos = _require_tensor(payload, "reference_pos")
    reference_atom_type = _require_tensor(payload, "reference_atom_type")
    reference_batch = _require_tensor(payload, "reference_batch")
    reference_node_attr = _require_tensor(payload, "reference_node_attr")

    pocket_pos = _require_tensor(payload, "pocket_pos")
    pocket_batch = _require_tensor(payload, "pocket_batch")

    _validate_pos_tensor("query_pos", query_pos)
    _validate_pos_tensor("reference_pos", reference_pos)
    _validate_pos_tensor("pocket_pos", pocket_pos)

    _validate_atom_type_tensor("query_atom_type", query_atom_type, query_pos)
    _validate_atom_type_tensor("reference_atom_type", reference_atom_type, reference_pos)

    _validate_batch_tensor("query_batch", query_batch, query_pos)
    _validate_batch_tensor("reference_batch", reference_batch, reference_pos)
    _validate_batch_tensor("pocket_batch", pocket_batch, pocket_pos)

    _validate_node_attr_tensor("query_node_attr", query_node_attr, query_pos)
    _validate_node_attr_tensor("reference_node_attr", reference_node_attr, reference_pos)

    if pocket_pos.numel() == 0:
        raise ValueError(
            "pocket_pos is empty. DiffAlign example inference payload should include "
            "real pocket coordinates from pocket.pdb shifted by reference_center."
        )

    _validate_reference_center_metadata(payload)
    
def validate_diffalign_example_train_payload(payload: dict[str, object]) -> None:
    """Validate a DiffAlign-example pseudo-training payload.

    The train payload must satisfy all inference payload conventions and additionally
    include target_query_pos with the same shape as query_pos.

    한 줄 요약:
        학습 페이로드가 추론 규약을 모두 만족하고 query_pos와 같은 형상의
        target_query_pos를 갖는지 검증한다.

    생성이유:
        flow matching 학습은 소스(query_pos)와 타깃(target_query_pos)의 형상이
        정확히 일치해야 한다. 추론 검증에 더해 타깃 정합성을 추가로 강제한다.

    역할:
        보조(11단계)에서 학습 페이로드의 정합성을 종합 검증한다.

    메커니즘:
        먼저 validate_diffalign_example_inference_payload로 공통 규약을 검증한다.
        그 후 _require_tensor로 query_pos와 target_query_pos를 가져오고,
        _validate_pos_tensor로 타깃 좌표 형상을 확인한 뒤 두 텐서의 형상이
        같지 않으면 ValueError를 발생시킨다.

    파라미터:
        payload (dict[str, object]): 검증할 학습 페이로드.

    반환:
        None. 문제가 있으면 예외를 발생시킨다.

    파이프라인 단계:
        11단계(외부 diffalign 연동) - 학습 페이로드 종합 검증.
    """

    validate_diffalign_example_inference_payload(payload)

    query_pos = _require_tensor(payload, "query_pos")
    target_query_pos = _require_tensor(payload, "target_query_pos")

    _validate_pos_tensor("target_query_pos", target_query_pos)

    if query_pos.shape != target_query_pos.shape:
        raise ValueError(
            "query_pos and target_query_pos must have identical shape for flow matching. "
            f"query_pos={tuple(query_pos.shape)}, "
            f"target_query_pos={tuple(target_query_pos.shape)}. "
            "Expected target_query_pos = query_sdf_conformer - reference_center."
        )