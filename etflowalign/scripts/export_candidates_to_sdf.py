"""Export candidate coordinates tensor to multi-mol SDF."""
from __future__ import annotations
import argparse, torch
from rdkit import Chem

def _load_first(path):
    mols=[m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    if not mols: raise ValueError('No mol')
    return mols[0]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-pt', required=True)
    p.add_argument('--template-sdf', required=True)
    p.add_argument('--output-sdf', required=True)
    a=p.parse_args()
    obj=torch.load(a.input_pt, map_location='cpu')
    cand=obj['candidates']
    mol=_load_first(a.template_sdf)
    w=Chem.SDWriter(a.output_sdf)
    for i in range(cand.shape[0]):
        m=Chem.Mol(mol)
        conf=m.GetConformer()
        for j in range(m.GetNumAtoms()):
            x,y,z=[float(v) for v in cand[i,j]]
            conf.SetAtomPosition(j, Chem.rdGeometry.Point3D(x,y,z))
        m.SetProp('_Name', f'cand_{i:03d}')
        w.write(m)
    w.close(); print(f'[export] saved: {a.output_sdf}')
if __name__=='__main__': main()
