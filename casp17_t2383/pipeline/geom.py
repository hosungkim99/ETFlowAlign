#!/usr/bin/env python3
"""
geom.py - 공용 기하/정렬 유틸 (numpy; SC-RMSD는 rdkit lazy-import).

- kabsch: 회전+이동
- align_to_ref: 참조 Cα에 정렬하는 R,t
- ligand_automorphisms: SMILES 대칭(자기동형) 순열 목록 (SC-RMSD용)
- sc_rmsd: 대칭보정 RMSD (자기동형 중 최소)
주의: 자기동형 순열은 ligand_concat의 원자 순서가 SMILES 원자 순서와 같을 때(order-match)
       바로 적용됨. 다르면 항등순열로 폴백(=plain RMSD) + 경고.
"""
import numpy as np


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def align_to_ref(ref_ca, ca):
    """공통 잔기 Cα로 ca→ref 정렬하는 (R,t). 공통<50이면 None."""
    common = [k for k in ref_ca if k in ca]
    if len(common) < 50:
        return None
    R, t = kabsch(np.array([ca[k] for k in common]),
                  np.array([ref_ca[k] for k in common]))
    return R, t


def apply_rt(R, t, coords):
    return (R @ np.asarray(coords).T).T + t


def ligand_automorphisms(smiles, n_atoms, max_perms=64):
    """
    SMILES heavy-atom 그래프의 자기동형 순열 목록. 원자 수가 안 맞으면 [identity].
    rdkit 없으면/실패하면 [identity].
    """
    ident = tuple(range(n_atoms))
    try:
        from rdkit import Chem
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
        if mol is None or mol.GetNumAtoms() != n_atoms:
            return [ident]
        matches = mol.GetSubstructMatches(mol, uniquify=False, maxMatches=max_perms)
        perms = [m for m in matches if len(m) == n_atoms]
        return perms or [ident]
    except Exception:
        return [ident]


def sc_rmsd(P, Q, perms=None):
    """
    대칭보정 RMSD. P,Q: (N,3), 같은 좌표계(이미 단백질 정렬됨) 가정.
    perms: ligand_automorphisms 결과. None이면 plain RMSD.
    """
    P = np.asarray(P); Q = np.asarray(Q)
    if perms is None or len(perms) == 1:
        return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
    best = 1e18
    for pm in perms:
        d = float(np.sqrt(((P - Q[list(pm)]) ** 2).sum(1).mean()))
        if d < best:
            best = d
    return best

# 이 파일은 import 전용 공용 라이브러리입니다 (단독 실행 X).
# 스텝들이 `import geom` 으로 사용 → 스텝들과 같은 폴더에 두어야 함.
