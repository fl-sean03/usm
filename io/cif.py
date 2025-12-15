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
a soft import. For now, keep MolSAIC self-contained.

Public API:
- load_cif(path: str, *, mol_label="XXXX", mol_index=1, mol_block_name=None) -> USM
- save_cif(usm: USM, path: str, *, data_block_name=None, spacegroup=None, wrap_frac=True) -> str
"""

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.lattice import frac_to_xyz, lattice_inverse, lattice_matrix, xyz_to_frac


_NUM_WITH_ESD_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?$")


def _strip_quotes(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        return t[1:-1]
    return t


def _parse_cif_number(val: Any, default: float = float("nan")) -> float:
    """
    Parse CIF numeric tokens, tolerating uncertainty syntax like "8.9138(12)".
    """
    if val is None:
        return float(default)
    s = str(val).strip()
    if not s or s in (".", "?"):
        return float(default)
    m = _NUM_WITH_ESD_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _cif_tokenize(lines: Iterable[str]) -> Iterator[str]:
    """
    Tokenize CIF text into a flat token stream.

    Supports:
    - comments starting with '#'
    - quoted strings with single or double quotes
    - semicolon-delimited text blocks (starting with ';' in column 1)
      which are returned as a single token (content excludes the delimiter lines)
    """
    it = iter(lines)
    for raw in it:
        if raw is None:
            continue

        # Comment-only line
        if raw.startswith("#"):
            continue

        # Semicolon-delimited multi-line text field (CIF)
        if raw.startswith(";"):
            buf: List[str] = []
            for nxt in it:
                if nxt.startswith(";"):
                    break
                buf.append(nxt.rstrip("\n"))
            yield "\n".join(buf)
            continue

        # Normal line: strip comments and split
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        # shlex handles quoted strings; CIF is close enough for our minimal needs
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            parts = line.split()

        for tok in parts:
            yield tok


@dataclass
class _CifLoop:
    tags: List[str]
    rows: List[List[str]]


def _parse_cif(tokens: Sequence[str]) -> Tuple[Dict[str, str], List[_CifLoop], Optional[str]]:
    """
    Parse a CIF token stream into:
    - data_items: scalar tag->value mapping (last occurrence wins)
    - loops: list of loops with tags + row values
    - data_block_name: last seen data_ name (or None)
    """
    data_items: Dict[str, str] = {}
    loops: List[_CifLoop] = []
    data_block_name: Optional[str] = None

    i = 0
    n = len(tokens)

    def _is_stop(tok: str) -> bool:
        tl = tok.lower()
        return tl == "loop_" or tl.startswith("data_") or tok.startswith("_")

    while i < n:
        tok = tokens[i]
        tl = tok.lower()

        if tl.startswith("data_"):
            data_block_name = tok[5:]
            i += 1
            continue

        if tl == "loop_":
            i += 1
            tags: List[str] = []
            while i < n and tokens[i].startswith("_"):
                tags.append(tokens[i])
                i += 1

            vals: List[str] = []
            while i < n and not _is_stop(tokens[i]):
                vals.append(tokens[i])
                i += 1

            if not tags:
                continue

            rows: List[List[str]] = []
            width = len(tags)
            if width > 0:
                for j in range(0, len(vals), width):
                    chunk = vals[j : j + width]
                    if len(chunk) != width:
                        break
                    rows.append([str(x) for x in chunk])

            loops.append(_CifLoop(tags=tags, rows=rows))
            continue

        if tok.startswith("_"):
            tag = tok
            i += 1
            if i < n and not _is_stop(tokens[i]):
                data_items[tag] = tokens[i]
                i += 1
            else:
                data_items[tag] = ""
            continue

        i += 1

    return data_items, loops, data_block_name


def _find_atom_site_loop(loops: List[_CifLoop]) -> Optional[_CifLoop]:
    """
    Find a CIF loop that contains fractional atom site coordinates.
    """
    needed = {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}
    for lp in loops:
        tags = {t.lower() for t in lp.tags}
        if needed.issubset(tags):
            return lp
    return None


def _infer_element_from_label(label: str) -> str:
    """
    Best-effort element inference from a CIF label like 'Zn1', 'C2', 'H1A'.
    """
    s = str(label).strip()
    if not s:
        return "X"
    m = re.match(r"^([A-Za-z]+)", s)
    if not m:
        return "X"
    sym = m.group(1)
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


def load_cif(
    path: str,
    *,
    mol_label: str = "XXXX",
    mol_index: int = 1,
    mol_block_name: Optional[str] = None,
    expand_symmetry: bool = False,
) -> USM:
    """
    Load a CIF into a USM object.

    Parameters:
      mol_label/mol_index: populate USM identity columns (CAR/MDF compatibility).
      mol_block_name: stored in atoms['mol_block_name'] (else derived from data_ block name when available).
      expand_symmetry: not implemented in v0.1; explicit flag to avoid silent behavior changes.

    Returns:
      USM with atoms populated and cell.pbc=True when cell params are present.
      Bonds are not inferred from CIF.
    """
    if expand_symmetry:
        raise NotImplementedError("CIF symmetry expansion is not implemented in this v0.1 loader.")

    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    tokens = list(_cif_tokenize(text))
    data_items, loops, data_name = _parse_cif(tokens)

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

    atoms_records: List[Dict[str, Any]] = []
    frac_rows: List[List[float]] = []

    for row in lp.rows:
        label = _strip_quotes(_get(row, "_atom_site_label"))
        if not label:
            continue

        type_sym = _strip_quotes(_get(row, "_atom_site_type_symbol"))
        element = type_sym if type_sym else _infer_element_from_label(label)

        fx = _parse_cif_number(_get(row, "_atom_site_fract_x"))
        fy = _parse_cif_number(_get(row, "_atom_site_fract_y"))
        fz = _parse_cif_number(_get(row, "_atom_site_fract_z"))

        if not np.isfinite([fx, fy, fz]).all():
            continue

        frac_rows.append([float(fx), float(fy), float(fz)])

        atoms_records.append(
            {
                "name": str(label),
                "element": str(element),
                # No forcefield assignment yet
                "atom_type": "xx",
                "charge": np.float32(0.0),
                # filled after frac conversion
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "mol_label": str(mol_label),
                "mol_index": int(mol_index),
                "mol_block_name": str(mol_block_name if mol_block_name is not None else (data_name or "")),
            }
        )

    if not atoms_records:
        raise ValueError(f"No atoms parsed from CIF atom_site loop: {p}")

    # Fractional -> Cartesian conversion
    A = lattice_matrix(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
    frac = np.asarray(frac_rows, dtype=np.float64)
    xyz = frac_to_xyz(A, frac)

    for i, rec in enumerate(atoms_records):
        rec["x"] = float(xyz[i, 0])
        rec["y"] = float(xyz[i, 1])
        rec["z"] = float(xyz[i, 2])

    atoms_df = pd.DataFrame(atoms_records)

    provenance = {
        "source_format": "cif",
        "source_path": str(p),
        "parse_notes": "",
    }
    preserved_text = {
        "cif_data_block": data_name or "",
        # Keep raw text for debugging/analysis
        "cif_lines": text,
    }

    usm = USM(atoms=atoms_df, bonds=None, molecules=None, cell=cell, provenance=provenance, preserved_text=preserved_text)
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