from __future__ import annotations

from typing import Optional, Any
from pathlib import Path
import numpy as np

from usm.core.model import USM


def _fmt_atom_name(name: str) -> str:
    # PDB atom name is columns 13-16; left/right justification rules depend on element,
    # but for simplicity we right-pad to 4 characters.
    n = (name or "X")[:4]
    return f"{n:>4s}"


def _fmt_res_name(res: str) -> str:
    r = (res or "RES")[:3]
    return f"{r:>3s}"


def _fmt_chain_id(chain: str) -> str:
    c = (chain or "A")
    return f"{c[:1]}"


def _fmt_element(elem: str) -> str:
    e = (elem or "X").upper()
    if len(e) == 1:
        return f" {e}"
    return e[:2]


def _float8_3(x: Any) -> str:
    try:
        f = float(x)
    except Exception:
        f = 0.0
    return f"{f:8.3f}"


def _float6_2(x: Any) -> str:
    try:
        f = float(x)
    except Exception:
        f = 0.0
    return f"{f:6.2f}"


def _compose_cryst1(cell: dict) -> Optional[str]:
    # CRYST1 a b c alpha beta gamma spacegroup
    if not bool(cell.get("pbc", False)):
        return None
    a = cell.get("a", np.nan)
    b = cell.get("b", np.nan)
    c = cell.get("c", np.nan)
    alpha = cell.get("alpha", np.nan)
    beta = cell.get("beta", np.nan)
    gamma = cell.get("gamma", np.nan)
    if not np.all(np.isfinite([a, b, c, alpha, beta, gamma])):
        return None
    spg = str(cell.get("spacegroup", ""))[:11]
    return (
        f"CRYST1{float(a):9.3f}{float(b):9.3f}{float(c):9.3f}"
        f"{float(alpha):7.2f}{float(beta):7.2f}{float(gamma):7.2f} {spg:<11s} 1"
    )


def save_pdb(usm: USM, path: str) -> str:
    """
    Minimal PDB writer:
      - Writes optional CRYST1 if cell is orthorhombic (angles present) and pbc True
      - Emits ATOM records with derived residue/chain if absent
      - Appends TER and END
    Column layout (fixed width):
      1-6   "ATOM  "
      7-11  serial (aid+1)
      13-16 atom name (right-aligned 4)
      17    altLoc (blank)
      18-20 resName (right-aligned 3)
      22    chainID (1)
      23-26 resSeq (right-aligned 4)
      27    iCode (blank)
      31-38 x (8.3)
      39-46 y (8.3)
      47-54 z (8.3)
      55-60 occupancy (6.2)
      61-66 tempFactor (6.2)
      77-78 element (right-aligned 2)
    """
    lines = []

    cryst1 = _compose_cryst1(usm.cell or {})
    if cryst1:
        lines.append(cryst1)

    atoms = usm.atoms.sort_values(by=["aid"]).reset_index(drop=True)

    # Derive residue fields
    # resName: mol_block_name or fallback "RES"
    # chainID: constant "A" unless 'chain_id' column exists
    # resSeq: use mol_index if available, else 1
    for idx, row in atoms.iterrows():
        serial = int(row.get("aid", idx)) + 1
        name = _fmt_atom_name(str(row.get("name", "X")))
        res_name = _fmt_res_name(str(row.get("mol_block_name", "")) if row.get("mol_block_name", "") else "RES")
        chain_id = _fmt_chain_id(str(row.get("chain_id", "A")) if "chain_id" in atoms.columns else "A")
        try:
            res_seq = int(row.get("mol_index", 1))
        except Exception:
            res_seq = 1
        x = _float8_3(row.get("x", 0.0))
        y = _float8_3(row.get("y", 0.0))
        z = _float8_3(row.get("z", 0.0))
        occ = _float6_2(row.get("occupancy", 1.00))
        tf = _float6_2(row.get("xray_temp_factor", 0.00))
        elem = _fmt_element(str(row.get("element", "X")))

        line = (
            f"ATOM  "
            f"{serial:5d} "
            f"{name}"
            f" "  # altLoc
            f"{res_name} "
            f"{chain_id}"
            f"{res_seq:4d}"
            f"    "  # iCode + 3 spaces
            f"{x}{y}{z}"
            f"{occ}{tf}"
            f"          "  # segID, charge not used
            f"{elem:>2s}"
        )
        lines.append(line)

    # Terminate last residue and file
    lines.append("TER")
    lines.append("END")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    return str(out_path)


__all__ = ["save_pdb"]