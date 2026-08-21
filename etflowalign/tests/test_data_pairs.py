"""로컬 검증: SDF -> payload -> collate -> 백본 forward 왕복.

RDKit 로 유사분자 2개를 만들어 SDF 로 쓰고, 텐서 계약 변환·배치·모델 입력까지
예외 없이 이어지는지 확인한다 (실데이터 없이 로컬에서 수초).

직접 실행: python -m etflowalign.tests.test_data_pairs
"""

from __future__ import annotations

import os
import tempfile

import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from etflowalign.backbone import EquivariantTransformer
from etflowalign.data.pairs import build_pair_payload, collate, tanimoto_2d, validate_payload


def _embed_to_sdf(smiles: str, path: str, seed: int = 0):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, f"임베딩 실패: {smiles}"
    AllChem.MMFFOptimizeMolecule(mol)
    w = Chem.SDWriter(path)
    w.write(mol)
    w.close()


def _make_pair(tmp: str):
    # 유사하지만 다른 두 분자 (query / reference)
    q_sdf = os.path.join(tmp, "query.sdf")
    r_sdf = os.path.join(tmp, "reference.sdf")
    _embed_to_sdf("c1ccccc1CCN", q_sdf, seed=1)      # phenethylamine
    _embed_to_sdf("c1ccccc1CCO", r_sdf, seed=2)      # phenethyl alcohol
    return q_sdf, r_sdf


def run_roundtrip():
    tmp = tempfile.mkdtemp(prefix="etfa_pairs_")
    q_sdf, r_sdf = _make_pair(tmp)

    p = build_pair_payload(q_sdf, r_sdf, mol_id="test0")
    validate_payload(p)

    q = Chem.MolFromSmiles("c1ccccc1CCN")
    r = Chem.MolFromSmiles("c1ccccc1CCO")
    tani = tanimoto_2d(Chem.AddHs(q), Chem.AddHs(r))

    batch = collate([p, p])   # 2개짜리 배치
    model = EquivariantTransformer(hidden_channels=32, num_layers=2, num_rbf=16,
                                   num_heads=2, cutoff=10.0)
    t = torch.rand(batch["num_graphs"])
    v = model(batch["z"], batch["pos"], t, batch["batch"])

    ok = v.shape == batch["pos"].shape and torch.isfinite(v).all()
    return {
        "Nq": int(p["z"].numel()),
        "Nr": int(p["ref_z"].numel()),
        "bonds": int(p["bonds"].size(1)),
        "tanimoto": tani,
        "v_shape": tuple(v.shape),
        "passed": bool(ok),
    }


def test_data_pairs_roundtrip():
    res = run_roundtrip()
    assert res["passed"], res


if __name__ == "__main__":
    res = run_roundtrip()
    print(f"[data] query 원자수 Nq={res['Nq']}, reference Nr={res['Nr']}, "
          f"query 결합수={res['bonds']}")
    print(f"[data] 2D Tanimoto(query,ref)={res['tanimoto']:.3f}")
    print(f"[data] collate -> 백본 forward 출력 {res['v_shape']}")
    print(f"[LOCAL DATA] {'PASS' if res['passed'] else 'FAIL'} — SDF->payload->batch->model 왕복 OK")
