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


def save_pdb(
    usm: USM,
    path: str,
    include_conect: bool = False,
    include_model: bool = False,
    model_index: int = 1,
    conect_policy: str = "dedup",
) -> str:
    """
    Minimal PDB writer with optional CONECT and MODEL/ENDMDL:
      - Writes optional CRYST1 if pbc True and all lattice params are finite (unchanged)
      - Emits ATOM records with derived residue/chain if absent (unchanged)
      - Appends TER and END
      - Optional CONECT records from usm.bonds (a1,a2 -> serials aid+1), up to 4 neighbors per line
      - Optional MODEL/ENDMDL framing (single model), with CRYST1 placed after MODEL when present

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

    Record order:
      - include_model=False: CRYST1 (if any), ATOM..., TER, [CONECT...], END
      - include_model=True:  MODEL {model_index}, CRYST1 (if any), ATOM..., TER, [CONECT...], ENDMDL, END

    Parameters:
      include_conect: emit CONECT records derived from usm.bonds (default False)
      include_model: emit MODEL/ENDMDL records around ATOM (default False)
      model_index: integer to write in MODEL line (default 1)
      conect_policy: "dedup" or "full"
         - "dedup": emit neighbors only on the lower-serial atom (each bond once)
         - "full":  emit neighbors on both atoms (both directions)
    """
    lines: list[str] = []

    atoms = usm.atoms.sort_values(by=["aid"]).reset_index(drop=True)

    def _emit_atoms(into: list[str]) -> None:
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
            into.append(line)

        # Terminate the last residue
        into.append("TER")

    def _compose_conect(into: list[str]) -> None:
        if not include_conect:
            return
        bonds = getattr(usm, "bonds", None)
        if bonds is None or len(bonds) == 0:
            return
        # Only include bonds between atoms present in this atoms table
        present_aids = set(atoms["aid"].astype(int).tolist()) if "aid" in atoms.columns else set(range(len(atoms)))
        policy = conect_policy if conect_policy in ("dedup", "full") else "dedup"

        # neighbors maps serial -> set of neighbor serials
        neighbors: dict[int, set[int]] = {}
        for _, br in bonds.iterrows():
            try:
                a1 = int(br["a1"])
                a2 = int(br["a2"])
            except Exception:
                continue
            if a1 == a2:
                continue
            if (a1 not in present_aids) or (a2 not in present_aids):
                continue
            s1 = a1 + 1
            s2 = a2 + 1
            if policy == "dedup":
                low = s1 if s1 < s2 else s2
                high = s2 if s1 < s2 else s1
                if low not in neighbors:
                    neighbors[low] = set()
                neighbors[low].add(high)
            else:  # "full"
                if s1 not in neighbors:
                    neighbors[s1] = set()
                if s2 not in neighbors:
                    neighbors[s2] = set()
                neighbors[s1].add(s2)
                neighbors[s2].add(s1)

        if not neighbors:
            return

        # Deterministic emission: serials ascending, neighbors ascending, 4 per line
        for s in sorted(neighbors.keys()):
            nbs = sorted(int(x) for x in neighbors[s])
            for i in range(0, len(nbs), 4):
                chunk = nbs[i : i + 4]
                line = "CONECT" + f"{s:5d}" + "".join(f"{n:5d}" for n in chunk)
                into.append(line)

    cryst1 = _compose_cryst1(usm.cell or {})

    if include_model:
        lines.append(f"MODEL {model_index:4d}")
        if cryst1:
            lines.append(cryst1)
        _emit_atoms(lines)
        _compose_conect(lines)
        lines.append("ENDMDL")
        lines.append("END")
    else:
        if cryst1:
            lines.append(cryst1)
        _emit_atoms(lines)
        _compose_conect(lines)
        lines.append("END")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    return str(out_path)


__all__ = ["save_pdb"]