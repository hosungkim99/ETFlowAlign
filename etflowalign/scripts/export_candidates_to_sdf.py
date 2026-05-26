from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
'''[역할]
1. diffalign_example_out.pt 로드
2. input_metadata.query_sdf 로드
3. candidates[k] 좌표를 query mol에 주입
4. reference_center_subtracted를 더해 global coordinate 복원
5. candidate_k.sdf 저장
'''
'''[목표]
diffalign_example_out.pt
        ↓
candidates tensor [K, Nq, 3]
        ↓
query_sdf RDKit Mol에 좌표 주입
        ↓
candidate_000.sdf, candidate_001.sdf 저장
'''

def load_pt(path: str | Path) -> dict[str, Any]:
    """Load an ETFlowAlign .pt output file.

    The file is produced by our own pipeline, so weights_only=False is used
    explicitly to avoid future-default ambiguity.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input .pt file not found: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload, got {type(obj)}")

    required = ["candidates", "scores", "metadata"]
    for key in required:
        if key not in obj:
            raise KeyError(f"Missing required key in output .pt: {key}")

    if not torch.is_tensor(obj["candidates"]):
        raise TypeError("obj['candidates'] must be a torch.Tensor")

    if obj["candidates"].ndim != 3 or obj["candidates"].shape[-1] != 3:
        raise ValueError(
            "obj['candidates'] must have shape [K, N, 3], "
            f"got {tuple(obj['candidates'].shape)}"
        )

    return obj


def load_query_mol(path: str | Path, *, sanitize: bool = True, remove_hs: bool = True) -> Chem.Mol:
    """Load query ligand template molecule from SDF/MOL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query molecule file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
        mol = supplier[0] if supplier is not None and len(supplier) > 0 else None
    else:
        raise ValueError(f"Unsupported query molecule extension: {path.suffix}")

    if mol is None:
        raise ValueError(f"Failed to load query molecule: {path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Query molecule has no conformer: {path}")

    return mol


def set_mol_positions(mol: Chem.Mol, coords: torch.Tensor) -> Chem.Mol:
    """Return a copy of mol with conformer positions set to coords[N,3]."""
    coords = coords.detach().cpu().float()

    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {tuple(coords.shape)}")

    if mol.GetNumAtoms() != coords.shape[0]:
        raise ValueError(
            "Atom count mismatch between query molecule and candidate coordinates: "
            f"mol atoms={mol.GetNumAtoms()}, coords atoms={coords.shape[0]}"
        )

    out = Chem.Mol(mol)

    if out.GetNumConformers() == 0:
        conf = Chem.Conformer(out.GetNumAtoms())
        out.AddConformer(conf, assignId=True)

    conf = out.GetConformer()

    for atom_idx in range(out.GetNumAtoms()):
        x, y, z = coords[atom_idx].tolist()
        conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))

    return out


def get_query_sdf_from_metadata(obj: dict[str, Any]) -> str | None:
    """Extract query_sdf path from ETFlowAlign output metadata."""
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    input_metadata = metadata.get("input_metadata", {})
    if not isinstance(input_metadata, dict):
        return None

    query_sdf = input_metadata.get("query_sdf")
    if query_sdf is None:
        return None

    return str(query_sdf)


def get_reference_center_from_metadata(obj: dict[str, Any]) -> torch.Tensor | None:
    """Extract reference center vector from ETFlowAlign output metadata."""
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    input_metadata = metadata.get("input_metadata", {})
    if not isinstance(input_metadata, dict):
        return None

    center = input_metadata.get("reference_center_subtracted")
    if center is None:
        return None

    if torch.is_tensor(center):
        center_tensor = center.detach().cpu().float()
    else:
        center_tensor = torch.tensor(center, dtype=torch.float32)

    if center_tensor.shape != (3,):
        raise ValueError(
            "reference_center_subtracted must have shape [3], "
            f"got {tuple(center_tensor.shape)}"
        )

    return center_tensor


def write_candidate_sdfs(
    *,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    query_mol: Chem.Mol,
    out_dir: str | Path,
    prefix: str = "candidate",
    restore_center: torch.Tensor | None = None,
    top_k: int | None = None,
    write_multisdf: bool = True,
) -> list[Path]:
    """Write candidate conformations to individual SDF files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidates.detach().cpu().float()

    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.detach().cpu().float().view(-1)

    num_candidates = candidates.shape[0]

    if scores.numel() != num_candidates:
        raise ValueError(
            f"scores length must match candidates K. scores={scores.numel()}, K={num_candidates}"
        )

    if top_k is None or top_k <= 0:
        selected_indices = list(range(num_candidates))
    else:
        selected_indices = list(range(min(int(top_k), num_candidates)))

    written_paths: list[Path] = []

    multi_writer = None
    multi_path = out_dir / f"{prefix}_all.sdf"
    if write_multisdf:
        multi_writer = Chem.SDWriter(str(multi_path))

    try:
        for rank, cand_idx in enumerate(selected_indices):
            coords = candidates[cand_idx]

            if restore_center is not None:
                coords = coords + restore_center.view(1, 3)

            mol = set_mol_positions(query_mol, coords)

            score_value = float(scores[cand_idx].item())
            mol.SetProp("_Name", f"{prefix}_{cand_idx:03d}")
            mol.SetProp("candidate_index", str(cand_idx))
            mol.SetProp("rank", str(rank))
            mol.SetProp("score", f"{score_value:.8f}")

            out_path = out_dir / f"{prefix}_{cand_idx:03d}.sdf"
            writer = Chem.SDWriter(str(out_path))
            writer.write(mol)
            writer.close()

            written_paths.append(out_path)

            if multi_writer is not None:
                multi_writer.write(mol)

    finally:
        if multi_writer is not None:
            multi_writer.close()
            written_paths.append(multi_path)

    return written_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ETFlowAlign candidates from .pt to SDF files."
    )

    parser.add_argument(
        "--input-pt",
        type=str,
        required=True,
        help="ETFlowAlign inference output .pt path.",
    )
    parser.add_argument(
        "--query-sdf",
        type=str,
        default="",
        help=(
            "Query SDF template path. If omitted, the script uses "
            "metadata['input_metadata']['query_sdf'] from the .pt file."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for SDF files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="candidate",
        help="Output SDF filename prefix.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Number of candidates to export. 0 means export all.",
    )
    parser.add_argument(
        "--no-restore-center",
        action="store_true",
        help=(
            "Do not add metadata['input_metadata']['reference_center_subtracted'] "
            "to candidate coordinates."
        ),
    )
    parser.add_argument(
        "--no-multisdf",
        action="store_true",
        help="Do not write a combined multi-molecule SDF.",
    )
    parser.add_argument(
        "--keep-hs",
        action="store_true",
        help="Keep hydrogens when loading query SDF. Default removes hydrogens.",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable RDKit sanitization when loading query SDF.",
    )

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    obj = load_pt(args.input_pt)

    query_sdf = args.query_sdf.strip()
    if not query_sdf:
        query_sdf = get_query_sdf_from_metadata(obj) or ""

    if not query_sdf:
        raise ValueError(
            "Query SDF path was not provided and could not be inferred from metadata. "
            "Pass --query-sdf explicitly."
        )

    query_mol = load_query_mol(
        query_sdf,
        sanitize=not args.no_sanitize,
        remove_hs=not args.keep_hs,
    )

    candidates = obj["candidates"]
    scores = obj["scores"]

    restore_center = None
    if not args.no_restore_center:
        restore_center = get_reference_center_from_metadata(obj)
        if restore_center is None:
            print("[export] reference_center_subtracted not found; exporting centered coordinates.")
        else:
            print(f"[export] restoring global coordinates with center={restore_center.tolist()}")
    else:
        print("[export] exporting centered coordinates without global restoration.")

    written = write_candidate_sdfs(
        candidates=candidates,
        scores=scores,
        query_mol=query_mol,
        out_dir=args.out_dir,
        prefix=args.prefix,
        restore_center=restore_center,
        top_k=args.top_k,
        write_multisdf=not args.no_multisdf,
    )

    print(f"[export] input_pt: {args.input_pt}")
    print(f"[export] query_sdf: {query_sdf}")
    print(f"[export] candidates: {tuple(candidates.shape)}")
    print(f"[export] scores: {tuple(scores.shape)}")
    print(f"[export] out_dir: {args.out_dir}")
    print("[export] written files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
    
'''[실행 예제 - global 복원 없이, 모델이 직접 낸 centered coordinate를 보고 싶다면]
python -m etflowalign.scripts.export_candidates_to_sdf \
  --input-pt /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_out.pt \
  --out-dir /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_sdf_centered \
  --prefix diffalign_example_candidate_centered \
  --no-restore-center
'''

'''[실행 예제]
 python -m etflowalign.scripts.export_candidates_to_sdf \
  --input-pt /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_out.pt \
  --out-dir /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_sdf \
  --prefix diffalign_example_candidate        
'''

