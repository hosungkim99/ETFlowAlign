"""Prepare a DiffAlign-style training .pt batch with target_query_pos + node attrs."""
from __future__ import annotations
import argparse, torch
from rdkit import Chem
from etflowalign.diffalign_adapter import make_train_payload

def _load_first_mol(path:str):
    mols=[m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    if not mols: raise ValueError(f'No molecule in {path}')
    return mols[0]

def _pos(m):
    c=m.GetConformer(); out=[]
    for i in range(m.GetNumAtoms()):
        p=c.GetAtomPosition(i); out.append([p.x,p.y,p.z])
    return torch.tensor(out, dtype=torch.float32)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--query-sdf', required=True)
    p.add_argument('--reference-sdf', required=True)
    p.add_argument('--target-query-sdf', required=True)
    p.add_argument('--output-pt', required=True)
    a=p.parse_args()
    q=_load_first_mol(a.query_sdf); r=_load_first_mol(a.reference_sdf); t=_load_first_mol(a.target_query_sdf)
    torch.save(make_train_payload(q,r,_pos(t)), a.output_pt)
    print(f'[prepare] saved: {a.output_pt}')
if __name__=='__main__': main()
