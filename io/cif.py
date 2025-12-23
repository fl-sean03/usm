from __future__ import annotations

"""
CIF import (v0.1)

Goal: load a crystallographic information file (CIF) into the USM data model.

Scope (intentionally minimal and dependency-free):
- Parse lattice parameters: a,b,c, alpha,beta,gamma
- Parse an atom_site loop containing:
    _atom_site_label
    _atom_site_type_symbol (preferred) or _atom_site_label used to infer element
    _atom_site_fract_x/_y/_z
  Additional columns are ignored.
- Convert fractional coordinates to Cartesian coordinates using USM lattice helpers.
- Do NOT expand symmetry operations. We import the atom sites as provided in the CIF.

This is sufficient for validating CIF-import against manually-exported Materials Studio CAR/MDF
for the CALF-20 example in assets/NIST.

If you need a full crystallography pipeline (symmetry expansion, disorder handling, etc.),
we should consider adding an optional third-party dependency (gemmi / ASE / pymatgen) behind
a soft import. For now, keep USM dependency-light.

Public API:
- load_cif(path: str, *, mol_label="XXXX", mol_index=1, mol_block_name=None) -> USM
- save_cif(usm: USM, path: str, *, data_block_name=None, spacegroup=None, wrap_frac=True) -> str
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.lattice import frac_to_xyz, lattice_inverse, lattice_matrix, xyz_to_frac, wrap_to_frac

# Import parsing internals from _cif_parser
from ._cif_parser import (
    _CifLoop,
    _cif_tokenize,
    _find_atom_site_loop,
    _infer_element_from_label,
    _parse_cif,
    _parse_cif_number,
    _parse_symop_string,
    _parse_symmetry_code,
    _strip_quotes,
)


def load_cif(
    path: str,
    *,
    mol_label: str = "XXXX",
    mol_index: int = 1,
    mol_block_name: Optional[str] = None,
    expand_symmetry: bool = False,
    sym_tol: float = 0.1,
) -> USM:
    """
    Load a CIF into a USM object.

    Parameters:
      mol_label/mol_index: populate USM identity columns (CAR/MDF compatibility).
      mol_block_name: stored in atoms['mol_block_name'] (else derived from data_ block name when available).
      expand_symmetry: if True, generate P1 unit cell using symmetry operations and occupancy filtering.
      sym_tol: Cartesian tolerance (Angstrom) for deduplicating atoms on special positions.

    Returns:
      USM with atoms and (if expand_symmetry=True and bonds present) bonds populated.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    tokens = list(_cif_tokenize(text))
    data_items, loops, data_name = _parse_cif(tokens)

    # Gather symmetry operators
    symops: List[Tuple[np.ndarray, np.ndarray]] = []
    symop_loop = None
    for lp in loops:
        if "_space_group_symop_operation_xyz" in [t.lower() for t in lp.tags]:
            symop_loop = lp
            break

    if symop_loop:
        tag_idx = [t.lower() for t in symop_loop.tags].index("_space_group_symop_operation_xyz")
        for row in symop_loop.rows:
            symops.append(_parse_symop_string(row[tag_idx]))

    # Fallback to P1 identity if no symops found
    if not symops:
        symops.append((np.eye(3), np.zeros(3)))

    # Cell parameters (required for meaningful cartesian conversion)
    a = _parse_cif_number(data_items.get("_cell_length_a"))
    b = _parse_cif_number(data_items.get("_cell_length_b"))
    c = _parse_cif_number(data_items.get("_cell_length_c"))
    alpha = _parse_cif_number(data_items.get("_cell_angle_alpha"), default=90.0)
    beta = _parse_cif_number(data_items.get("_cell_angle_beta"), default=90.0)
    gamma = _parse_cif_number(data_items.get("_cell_angle_gamma"), default=90.0)

    spacegroup = _strip_quotes(
        data_items.get("_space_group_name_H-M_alt") or data_items.get("_space_group_name_Hall") or ""
    )

    cell = dict(
        pbc=True,
        a=float(a),
        b=float(b),
        c=float(c),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        spacegroup=str(spacegroup),
    )

    lp = _find_atom_site_loop(loops)
    if lp is None:
        raise ValueError(f"No atom_site loop with fract_x/y/z found in CIF: {p}")

    tag_to_i = {t.lower(): i for i, t in enumerate(lp.tags)}

    def _get(row: List[str], tag: str) -> str:
        idx = tag_to_i.get(tag.lower())
        return "" if idx is None or idx >= len(row) else str(row[idx])

    A = lattice_matrix(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))

    au_atoms: List[Dict[str, Any]] = []
    for row in lp.rows:
        label = _strip_quotes(_get(row, "_atom_site_label"))
        if not label:
            continue

        occupancy = _parse_cif_number(_get(row, "_atom_site_occupancy"), default=1.0)
        # Rule D1: skip low occupancy (unless tagged framework, but we default to no for solvent)
        if expand_symmetry and occupancy < 1.0:
            continue

        fx = _parse_cif_number(_get(row, "_atom_site_fract_x"))
        fy = _parse_cif_number(_get(row, "_atom_site_fract_y"))
        fz = _parse_cif_number(_get(row, "_atom_site_fract_z"))

        if not np.isfinite([fx, fy, fz]).all():
            continue

        type_sym = _strip_quotes(_get(row, "_atom_site_type_symbol"))
        element = type_sym if type_sym else _infer_element_from_label(label)

        au_atoms.append(
            {"label": label, "element": element, "frac": np.array([fx, fy, fz], dtype=np.float64), "occupancy": occupancy}
        )

    if not au_atoms:
        raise ValueError(f"No atoms parsed from CIF atom_site loop: {p}")

    atoms_records: List[Dict[str, Any]] = []
    # Map from (au_idx, op_idx) -> p1_idx for bond resolution
    au_op_to_p1: Dict[Tuple[int, int], int] = {}

    if expand_symmetry:
        p1_fracs: List[np.ndarray] = []
        for i_au, au_atom in enumerate(au_atoms):
            for i_op, (rot, trans) in enumerate(symops):
                f_new = rot @ au_atom["frac"] + trans
                f_wrapped = wrap_to_frac(f_new)

                # Deduplicate based on cartesian distance
                is_duplicate = False
                pos_cart = frac_to_xyz(A, f_wrapped.reshape(1, 3))[0]
                for i_p1, existing_f in enumerate(p1_fracs):
                    existing_cart = frac_to_xyz(A, existing_f.reshape(1, 3))[0]
                    # Since we wrapped to [0,1), standard dist check is sufficient for unit cell deduplication.
                    dist = np.linalg.norm(pos_cart - existing_cart)
                    if dist < sym_tol:
                        is_duplicate = True
                        au_op_to_p1[(i_au, i_op)] = i_p1
                        break

                if not is_duplicate:
                    p1_idx = len(atoms_records)
                    p1_fracs.append(f_wrapped)
                    au_op_to_p1[(i_au, i_op)] = p1_idx

                    xyz = frac_to_xyz(A, f_wrapped.reshape(1, 3))[0]
                    atoms_records.append(
                        {
                            "name": f"{au_atom['label']}_{i_op+1}",
                            "element": au_atom["element"],
                            "atom_type": "xx",
                            "charge": np.float32(0.0),
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "mol_label": str(mol_label),
                            "mol_index": int(mol_index),
                            "mol_block_name": str(mol_block_name if mol_block_name is not None else (data_name or "")),
                            "occupancy": float(au_atom["occupancy"]),
                        }
                    )
    else:
        for i_au, au_atom in enumerate(au_atoms):
            xyz = frac_to_xyz(A, au_atom["frac"].reshape(1, 3))[0]
            atoms_records.append(
                {
                    "name": au_atom["label"],
                    "element": au_atom["element"],
                    "atom_type": "xx",
                    "charge": np.float32(0.0),
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                    "mol_label": str(mol_label),
                    "mol_index": int(mol_index),
                    "mol_block_name": str(mol_block_name if mol_block_name is not None else (data_name or "")),
                    "occupancy": float(au_atom["occupancy"]),
                }
            )

    atoms_df = pd.DataFrame(atoms_records)

    # Bond Perception / Parsing
    bonds_records: List[Dict[str, Any]] = []
    bond_loop = None
    for lp in loops:
        tags = {t.lower() for t in lp.tags}
        if "_geom_bond_atom_site_label_1" in tags and "_geom_bond_atom_site_label_2" in tags:
            bond_loop = lp
            break

    if bond_loop:
        btags = [t.lower() for t in bond_loop.tags]
        idx1 = btags.index("_geom_bond_atom_site_label_1")
        idx2 = btags.index("_geom_bond_atom_site_label_2")
        idx_sym = btags.index("_geom_bond_site_symmetry_2") if "_geom_bond_site_symmetry_2" in btags else -1

        label_to_au_idx = {a["label"]: i for i, a in enumerate(au_atoms)}

        for brow in bond_loop.rows:
            l1, l2 = _strip_quotes(brow[idx1]), _strip_quotes(brow[idx2])
            if l1 not in label_to_au_idx or l2 not in label_to_au_idx:
                continue

            au_idx1 = label_to_au_idx[l1]
            au_idx2 = label_to_au_idx[l2]
            sym_code = _strip_quotes(brow[idx_sym]) if idx_sym >= 0 else "."
            op_idx_rel, t_rel = _parse_symmetry_code(sym_code)

            if expand_symmetry:
                # For each operator k, create the expanded bond
                for i_op1, (rot1, trans1) in enumerate(symops):
                    # Atom 1 in P1
                    f1_raw = rot1 @ au_atoms[au_idx1]["frac"] + trans1
                    f1_wrapped = wrap_to_frac(f1_raw)
                    s1 = np.floor(f1_raw + 1e-8).astype(np.int32)

                    # Atom 2 in P1
                    # The CIF bond is between AU1 and M_rel(AU2) + T_rel
                    # Applying M1 to both: M1(AU1) and M1(M_rel(AU2) + T_rel)
                    rot_rel, trans_rel = symops[op_idx_rel]
                    f2_raw = rot1 @ (rot_rel @ au_atoms[au_idx2]["frac"] + trans_rel + t_rel) + trans1
                    f2_wrapped = wrap_to_frac(f2_raw)
                    s2 = np.floor(f2_raw + 1e-8).astype(np.int32)

                    # Find P1 indices
                    p1_idx1 = au_op_to_p1.get((au_idx1, i_op1), -1)
                    p1_idx2 = -1

                    # For p1_idx2, search p1_fracs
                    pos2_cart = frac_to_xyz(A, f2_wrapped.reshape(1, 3))[0]
                    # Note: p1_fracs is not accessible here, let's use atoms_records for coords
                    for i_p1, rec in enumerate(atoms_records):
                        existing_cart = np.array([rec["x"], rec["y"], rec["z"]], dtype=np.float64)
                        if np.linalg.norm(pos2_cart - existing_cart) < sym_tol:
                            p1_idx2 = i_p1
                            break

                    if p1_idx1 >= 0 and p1_idx2 >= 0:
                        shift = s2 - s1
                        bonds_records.append(
                            {
                                "a1": int(p1_idx1),
                                "a2": int(p1_idx2),
                                "ix": int(shift[0]),
                                "iy": int(shift[1]),
                                "iz": int(shift[2]),
                                "order": 1.0,
                                "type": "single",
                            }
                        )
            else:
                # Minimal load: just AU bond
                bonds_records.append(
                    {
                        "a1": int(au_idx1),
                        "a2": int(au_idx2),
                        "ix": int(t_rel[0]),
                        "iy": int(t_rel[1]),
                        "iz": int(t_rel[2]),
                        "order": 1.0,
                        "type": "single",
                    }
                )

    # Deduplicate bonds
    if bonds_records:
        unique_bonds = {}
        for b in bonds_records:
            # Canonicalize
            a1, a2 = b["a1"], b["a2"]
            ix, iy, iz = b["ix"], b["iy"], b["iz"]
            if a1 > a2:
                a1, a2 = a2, a1
                ix, iy, iz = -ix, -iy, -iz
            elif a1 == a2:
                # Self bond: ensure lexicographically positive shift
                if ix < 0 or (ix == 0 and iy < 0) or (ix == 0 and iy == 0 and iz < 0):
                    ix, iy, iz = -ix, -iy, -iz

            key = (a1, a2, ix, iy, iz)
            if key not in unique_bonds:
                unique_bonds[key] = b
        bonds_df = pd.DataFrame(list(unique_bonds.values()))
    else:
        bonds_df = None

    provenance = {
        "source_format": "cif",
        "source_path": str(p),
        "parse_notes": f"expand_symmetry={expand_symmetry}",
    }
    preserved_text = {
        "cif_data_block": data_name or "",
        # Keep raw text for debugging/analysis
        "cif_lines": text,
    }

    usm = USM(
        atoms=atoms_df, bonds=bonds_df, molecules=None, cell=cell, provenance=provenance, preserved_text=preserved_text
    )
    usm.validate_basic()
    return usm


def _format_cif_float(x: float, prec: int = 10) -> str:
    """
    CIF is generally tolerant of float formatting; keep it stable and reasonably precise.
    """
    try:
        f = float(x)
    except Exception:
        return "?"
    if not np.isfinite(f):
        return "?"
    # Avoid scientific notation for typical cell/frac values; fall back if needed.
    s = f"{f:.{prec}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _wrap01(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates to [0, 1)."""
    return frac - np.floor(frac)


def save_cif(
    usm: USM,
    path: str,
    *,
    data_block_name: str | None = None,
    spacegroup: str | None = None,
    wrap_frac: bool = True,
) -> str:
    """
    Save a USM instance to a minimal CIF.

    Notes / scope:
    - Writes cell parameters from usm.cell (requires finite a,b,c,alpha,beta,gamma).
    - Writes a single atom_site loop with:
        _atom_site_label, _atom_site_type_symbol, _atom_site_fract_x/y/z
    - Bonds/connectivity are not represented (CIF does not carry MDF-style topology by default).
    - Symmetry operations are not written; we write P1 unless a spacegroup string is provided.

    Parameters:
      data_block_name: CIF 'data_' block name. Defaults to preserved cif_data_block, then 'USM'.
      spacegroup: overrides usm.cell['spacegroup'].
      wrap_frac: if True, wrap fractional coords into [0,1) before writing.

    Returns:
      The output path.
    """
    cell = dict(usm.cell or {})
    req = ["a", "b", "c", "alpha", "beta", "gamma"]
    vals = [cell.get(k, np.nan) for k in req]
    if not np.isfinite(np.asarray(vals, dtype=np.float64)).all():
        raise ValueError("Cannot save CIF: USM.cell must contain finite a,b,c,alpha,beta,gamma")

    a, b, c, alpha, beta, gamma = [float(v) for v in vals]

    sg = spacegroup
    if sg is None:
        sg = str(cell.get("spacegroup") or "").strip()

    db = data_block_name
    if db is None:
        db = str((usm.preserved_text or {}).get("cif_data_block") or "").strip() or "USM"

    A = lattice_matrix(a, b, c, alpha, beta, gamma)
    Ainv = lattice_inverse(A)

    atoms = usm.atoms.sort_values(by=["aid"]).reset_index(drop=True)
    xyz = atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
    frac = xyz_to_frac(Ainv, xyz)
    if wrap_frac:
        frac = _wrap01(frac)

    lines: list[str] = []
    lines.append(f"data_{db}")
    lines.append(f"_cell_length_a    {_format_cif_float(a, prec=6)}")
    lines.append(f"_cell_length_b    {_format_cif_float(b, prec=6)}")
    lines.append(f"_cell_length_c    {_format_cif_float(c, prec=6)}")
    lines.append(f"_cell_angle_alpha {_format_cif_float(alpha, prec=6)}")
    lines.append(f"_cell_angle_beta  {_format_cif_float(beta, prec=6)}")
    lines.append(f"_cell_angle_gamma {_format_cif_float(gamma, prec=6)}")
    if sg:
        # Quote for safety (spaces, minus signs, etc.)
        lines.append(f"_space_group_name_H-M_alt '{sg}'")
    else:
        lines.append("_space_group_name_H-M_alt 'P 1'")
    lines.append("")
    lines.append("loop_")
    lines.append("_atom_site_label")
    lines.append("_atom_site_type_symbol")
    lines.append("_atom_site_fract_x")
    lines.append("_atom_site_fract_y")
    lines.append("_atom_site_fract_z")

    # Write rows
    for i, row in atoms.iterrows():
        label = str(row.get("name") or "").strip()
        if not label:
            # CIF labels must be non-empty; synthesize something stable
            el = str(row.get("element") or "X").strip() or "X"
            label = f"{el}{int(i) + 1}"
        el = str(row.get("element") or "").strip()
        if not el:
            el = _infer_element_from_label(label)

        fx, fy, fz = frac[i, 0], frac[i, 1], frac[i, 2]
        lines.append(
            f"{label} {el} "
            f"{_format_cif_float(fx, prec=10)} "
            f"{_format_cif_float(fy, prec=10)} "
            f"{_format_cif_float(fz, prec=10)}"
        )

    Path(path).write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    return path


__all__ = ["load_cif", "save_cif"]
