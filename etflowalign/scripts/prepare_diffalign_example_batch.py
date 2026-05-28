"""Prepare a DiffAlign-style inference .pt batch with node attributes."""
from __future__ import annotations
import argparse, torch
from rdkit import Chem
from etflowalign.diffalign_adapter import make_inference_payload

def _load_first_mol(path:str):
    if path.endswith('.sdf'):
        mols=[m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
        if not mols: raise ValueError(f'No molecule in {path}')
        return mols[0]
    raise ValueError('Only .sdf supported')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--query-sdf', required=True)
    p.add_argument('--reference-sdf', required=True)
    p.add_argument('--output-pt', required=True)
    a=p.parse_args()
    q=_load_first_mol(a.query_sdf); r=_load_first_mol(a.reference_sdf)
    torch.save(make_inference_payload(q,r), a.output_pt)
    print(f'[prepare] saved: {a.output_pt}')
if __name__=='__main__': main()
