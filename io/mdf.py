from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from usm.core.model import USM


# Regex to parse an MDF atom/topology line into named fields.
# Example (WAT):
# XXXX_1:H1           H  h*      ?     0  0     0.4100 0 0 8 1.0000  0.0000 O1
# Example (DOP, with formal charge '1+'):
# MOL2_1:N1           N  NH1     ?     0  1+   -0.1000 0 0 8 1.0000  0.0000 H8 C8 H9 H12
MDF_LINE_RE = re.compile(
    r"""^\s*
    (?P<prefix>\S+)                                        # e.g., MOL2_1:C1
    \s+
    (?P<element>\S+)\s+
    (?P<atom_type>\S+)\s+
    (?P<charge_group>\S+)\s+
    (?P<isotope>\S+)\s+
    (?P<formal_charge>\S+)\s+
    (?P<charge>[-+]?\d+(?:\.\d+)?)\s+
    (?P<switching_atom>-?\d+)\s+
    (?P<oop_flag>-?\d+)\s+
    (?P<chirality_flag>-?\d+)\s+
    (?P<occupancy>[-+]?\d+(?:\.\d+)?)\s+
    (?P<xray_temp_factor>[-+]?\d+(?:\.\d+)?)
    (?:\s+(?P<connections>.*))?
    \s*$
    """,
    re.VERBOSE,
)

# Prefix pattern e.g. "MOL2_1:C1" or "XXXX_1:H1"
PREFIX_RE = re.compile(r"^(?P<mol_label>[A-Za-z0-9]+)_(?P<mol_index>\d+):(?P<name>\S+)$")


def _split_sections(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split MDF file into header lines, atom/topology lines, and footer lines.
    We consider the first line that matches MDF_LINE_RE as the start of atoms section,
    and "#end" (case-insensitive) as a footer marker thereafter.
    """
    header: List[str] = []
    atoms: List[str] = []
    footer: List[str] = []

    mode = "header"
    for ln in lines:
        s = ln.rstrip("\n")
        if mode == "header":
            if MDF_LINE_RE.match(s):
                mode = "atoms"
                atoms.append(s)
            else:
                header.append(s)
        elif mode == "atoms":
            if s.strip().lower().startswith("#end"):
                mode = "footer"
                footer.append(s)
            else:
                atoms.append(s)
        else:
            footer.append(s)
    return header, atoms, footer


def _current_molecule_name_from_header(header_lines: List[str]) -> Optional[str]:
    """
    For simple single-@molecule files, capture the latest @molecule name.
    If multiple @molecule sections exist, we keep order in preserved_text and still
    assign the last seen name to atoms following it (v0.1).
    """
    mol_name: Optional[str] = None
    for h in header_lines:
        hs = h.strip()
        if hs.lower().startswith("@molecule"):
            # Format: @molecule Name (name may contain spaces but examples show single token)
            parts = hs.split(maxsplit=1)
            if len(parts) == 2:
                mol_name = parts[1].strip()
            else:
                mol_name = ""
    return mol_name


def _molecule_order(header_lines: List[str]) -> List[str]:
    order: List[str] = []
    for h in header_lines:
        hs = h.strip()
        if hs.lower().startswith("@molecule"):
            parts = hs.split(maxsplit=1)
            if len(parts) == 2:
                order.append(parts[1].strip())
            else:
                order.append("")
    return order


def _parse_atom_line(line: str, default_mol_block_name: Optional[str]) -> Dict[str, Any]:
    m = MDF_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Unrecognized MDF atom line: {line}")
    gd = m.groupdict()

    # Parse the prefix into molecule label/index and atom name
    pm = PREFIX_RE.match(gd["prefix"])
    if not pm:
        raise ValueError(f"Unrecognized MDF atom prefix: {gd['prefix']}")
    mol_label = pm.group("mol_label")
    mol_index = int(pm.group("mol_index"))
    name = pm.group("name")

    conn = gd.get("connections") or ""
    # normalize spacing minimally (preserve token text order)
    connections_raw = conn.rstrip()

    rec: Dict[str, Any] = {
        "name": name,
        "element": gd["element"],
        "atom_type": gd["atom_type"],
        "charge_group": gd["charge_group"],
        "isotope": gd["isotope"],
        "formal_charge": gd["formal_charge"],  # keep string (can be "1+")
        "charge": float(gd["charge"]),
        "switching_atom": int(gd["switching_atom"]),
        "oop_flag": int(gd["oop_flag"]),
        "chirality_flag": int(gd["chirality_flag"]),
        "occupancy": float(gd["occupancy"]),
        "xray_temp_factor": float(gd["xray_temp_factor"]),
        "connections_raw": connections_raw if connections_raw else pd.NA,
        # Common USM columns
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "mol_label": mol_label,
        "mol_index": mol_index,
        "mol_block_name": default_mol_block_name or "",
    }
    return rec


def _build_bonds_from_connections(atoms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert connections_raw into a normalized undirected bonds table:

    Token grammar supported:
      - Simple name within same label/index scope: target[/order]            e.g., H/1.0
      - Fully qualified target: LABEL_INDEX:NAME[/order]                     e.g., XXXX_1729:H
      - Optional Materials Studio suffixes after '%' are ignored             e.g., XXXX_825:C%0-10#1

    Rules:
      - order defaults to 1.0 when omitted
      - If token contains LABEL_INDEX:NAME, use that label/index; otherwise use the source atom's (mol_label, mol_index)
      - Deduplicate by sorted (a1, a2)
    """
    # Map (mol_label, mol_index, name) -> aid
    key_to_aid: Dict[Tuple[str, int, str], int] = {}
    for _, r in atoms_df[["aid", "mol_label", "mol_index", "name"]].iterrows():
        key = (str(r["mol_label"]), int(r["mol_index"]), str(r["name"]))
        key_to_aid[key] = int(r["aid"])

    tgt_prefix_re = re.compile(r"^(?P<label>[A-Za-z0-9]+)_(?P<idx>\d+):(?P<name>\S+)$")

    def parse_target_token(tok: str, src_label: str, src_index: int) -> Tuple[str, int, str, Optional[float], Optional[str]]:
        # Split order part if present
        order_val: Optional[float] = None
        order_raw: Optional[str] = None
        base = tok
        if "/" in tok:
            base, order_raw = tok.split("/", 1)
            try:
                order_val = float(order_raw)
            except Exception:
                order_val = None
        # Strip any Materials Studio constraint suffix after '%' on the base token
        if "%" in base:
            base = base.split("%", 1)[0]

        label = src_label
        idx = src_index
        name = base
        # If fully qualified target is provided, use it
        m = tgt_prefix_re.match(base)
        if m:
            label = m.group("label")
            idx = int(m.group("idx"))
            name = m.group("name")
        return label, idx, name, order_val, order_raw

    bonds: List[Dict[str, Any]] = []
    seen = set()

    for _, r in atoms_df.iterrows():
        src_aid = int(r["aid"])
        src_label = str(r["mol_label"])
        src_index = int(r["mol_index"])
        raw = r.get("connections_raw")
        if pd.isna(raw) or not str(raw).strip():
            continue
        tokens = str(raw).split()
        for tok in tokens:
            t_label, t_idx, t_name, order_val, order_raw = parse_target_token(tok, src_label, src_index)
            key = (t_label, t_idx, t_name)
            tgt_aid = key_to_aid.get(key)
            if tgt_aid is None:
                # Could not resolve token; skip
                continue
            a1, a2 = (src_aid, int(tgt_aid))
            if a1 == a2:
                continue
            if a2 < a1:
                a1, a2 = a2, a1
            pair = (a1, a2)
            if pair in seen:
                continue
            seen.add(pair)
            bonds.append(
                {
                    "a1": a1,
                    "a2": a2,
                    "order": float(order_val) if order_val is not None else float(1.0),
                    "order_raw": order_raw if order_raw is not None else pd.NA,
                    "type": pd.NA,
                    "source": "mdf.connections",
                    "mol_index": src_index,
                    "notes": pd.NA,
                }
            )

    if not bonds:
        return pd.DataFrame(columns=["a1", "a2", "order", "order_raw", "type", "source", "mol_index", "notes"])
    bonds_df = pd.DataFrame(bonds)
    # bid and a1<a2 normalization handled by USM constructor
    return bonds_df


def load_mdf(path: str) -> USM:
    """
    Load a Materials Studio MDF file into a USM instance, parsing topology and connections.

    Notes:
    - Coordinates are not present in MDF; x,y,z remain NaN (to be populated from CAR or other sources).
    - connections_raw is preserved as text; bonds are inferred best-effort and deduplicated.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header, atom_lines, footer = _split_sections(lines)
    mol_block_name = _current_molecule_name_from_header(header)
    mol_order = _molecule_order(header)

    date_line = ""
    for h in header:
        if h.strip().lower().startswith("!date"):
            date_line = h

    atom_records: List[Dict[str, Any]] = []
    for ln in atom_lines:
        s = ln.strip()
        if not s:
            continue
        # Tolerate comments and non-atom lines within atom block (e.g., "!")
        if s.startswith("!"):
            continue
        if MDF_LINE_RE.match(ln) is None:
            continue
        atom_records.append(_parse_atom_line(ln, mol_block_name))

    atoms_df = pd.DataFrame(atom_records)
    # USM will add 'aid' and dtype-normalize
    usm = USM(
        atoms=atoms_df,
        bonds=None,
        molecules=None,
        cell=dict(pbc=False, a=np.nan, b=np.nan, c=np.nan, alpha=np.nan, beta=np.nan, gamma=np.nan, spacegroup=""),
        provenance={
            "source_format": "mdf",
            "source_path": str(p),
            "date_line": date_line,
            "parse_notes": "",
        },
        preserved_text={
            "mdf_header_lines": header,
            "mdf_molecule_order": mol_order,
            "mdf_footer_lines": footer,
        },
    )

    # Build bonds from connections_raw
    bonds_df = _build_bonds_from_connections(usm.atoms)
    if len(bonds_df) > 0:
        usm.bonds = bonds_df

    usm.validate_basic()
    return usm



def _format_float_mdf(val: Any, prec: int = 4) -> str:
    try:
        f = float(val)
        if np.isnan(f):
            f = 0.0
        return f"{f:.{prec}f}"
    except Exception:
        return f"{0.0:.{prec}f}"


def _order_token(order: Optional[float]) -> Optional[str]:
    if order is None or np.isclose(float(order), 1.0):
        return None
    # Minimal decimal representation (e.g., 1.5 instead of 1.5000)
    return f"{float(order):g}"


def _compose_connections_for_atom(
    row: pd.Series,
    aid_to_info: Dict[int, Tuple[str, int, str]],
    adj_list: Dict[int, List[Tuple[int, float]]],
    write_normalized_connections: bool
) -> str:
    # Prefer raw connections when available and lossless mode
    if not write_normalized_connections:
        raw = row.get("connections_raw")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # Current atom identity
    aid = int(row["aid"])
    neighbors = adj_list.get(aid)
    if not neighbors:
        return ""

    src_label = str(row.get("mol_label"))
    try:
        src_index = int(row.get("mol_index"))
    except Exception:
        src_index = 1

    tokens: list[str] = []
    for other_aid, order_val in neighbors:
        info = aid_to_info.get(other_aid)
        if not info:
            continue
        o_label, o_index, o_name = info
        
        same_mol = (o_label == src_label) and (int(o_index) == int(src_index))
        token_base = o_name if same_mol else f"{o_label}_{int(o_index)}:{o_name}"
        
        ord_token = _order_token(order_val)
        tokens.append(f"{token_base}/{ord_token}" if ord_token else token_base)

    return " ".join(tokens)


def save_mdf(usm: USM, path: str, preserve_headers: bool = True, write_normalized_connections: bool = False) -> str:
    """
    Save a USM instance to an MDF file.

    - If preserve_headers and preserved header/footer exist, write them byte-for-byte.
    - Otherwise, synthesize a canonical header with @column block and a single @molecule using the first mol_block_name.
    - Write atom lines with MDF's 12 columns and connections (raw or normalized).
    """
    from datetime import datetime

    header_lines = None
    footer_lines = None
    if preserve_headers:
        header_lines = (usm.preserved_text or {}).get("mdf_header_lines")
        footer_lines = (usm.preserved_text or {}).get("mdf_footer_lines")

    lines: list[str] = []

    if header_lines:
        lines.extend(header_lines)
    else:
        # Synthesize minimal header
        lines.append("!BIOSYM molecular_data 4")
        lines.append("")
        date_line = (usm.provenance or {}).get("date_line", "")
        if date_line and date_line.strip().lower().startswith("!date"):
            lines.append(date_line)
        else:
            lines.append("!Date: " + datetime.now().strftime("%a %b %d %H:%M:%S %Y") + "   USM Generated MDF file")
        lines.append("")
        lines.append("#topology")
        lines.append("")
        # Canonical @column block (1..12)
        lines.append("@column 1 element")
        lines.append("@column 2 atom_type")
        lines.append("@column 3 charge_group")
        lines.append("@column 4 isotope")
        lines.append("@column 5 formal_charge")
        lines.append("@column 6 charge")
        lines.append("@column 7 switching_atom")
        lines.append("@column 8 oop_flag")
        lines.append("@column 9 chirality_flag")
        lines.append("@column 10 occupancy")
        lines.append("@column 11 xray_temp_factor")
        lines.append("@column 12 connections")
        lines.append("")
        # @molecule: prefer preserved name from atoms if consistent
        mol_name = ""
        if "mol_block_name" in usm.atoms.columns:
            vals = usm.atoms["mol_block_name"].dropna().unique().tolist()
            if len(vals) == 1:
                mol_name = str(vals[0])
        if not mol_name:
            mol_name = "USM_Molecule"
        lines.append(f"@molecule {mol_name}")
        lines.append("")

    # Atom lines in aid order
    atoms = usm.atoms.sort_values(by=["aid"]).reset_index(drop=True)

    # Precompute maps for O(N) performance
    # aid -> (label, index, name)
    try:
        aid_vec = atoms["aid"].to_numpy().astype(int)
    except Exception:
        aid_vec = atoms["aid"].astype(int).to_numpy()
    
    label_list = atoms["mol_label"].astype("string").fillna("XXXX").astype(str).tolist()
    try:
        index_vec = atoms["mol_index"].to_numpy().astype(int)
    except Exception:
        index_vec = pd.to_numeric(atoms["mol_index"], errors="coerce").fillna(1).astype(int).to_numpy()
    name_list = atoms["name"].astype("string").fillna("X").astype(str).tolist()
    
    aid_to_info = {int(a): (label_list[i], int(index_vec[i]), name_list[i]) for i, a in enumerate(aid_vec)}

    # Precompute adjacency list
    adj_list: Dict[int, List[Tuple[int, float]]] = {}
    if usm.bonds is not None and len(usm.bonds) > 0:
        for _, br in usm.bonds.iterrows():
            a1 = int(br["a1"])
            a2 = int(br["a2"])
            order = float(br.get("order", 1.0))
            if a1 not in adj_list: adj_list[a1] = []
            if a2 not in adj_list: adj_list[a2] = []
            adj_list[a1].append((a2, order))
            adj_list[a2].append((a1, order))

    for _, row in atoms.iterrows():
        mol_label = "XXXX" if pd.isna(row.get("mol_label")) else str(row.get("mol_label"))
        _mol_index_val = row.get("mol_index")
        mol_index = 1 if pd.isna(_mol_index_val) else int(_mol_index_val)
        name = "X" if pd.isna(row.get("name")) else str(row.get("name"))
        prefix = f"{mol_label}_{mol_index}:{name}"

        element = "?" if pd.isna(row.get("element")) else str(row.get("element"))
        atom_type = "?" if pd.isna(row.get("atom_type")) else str(row.get("atom_type"))
        charge_group = "?" if pd.isna(row.get("charge_group")) else str(row.get("charge_group"))
        isotope = "?" if pd.isna(row.get("isotope")) else str(row.get("isotope"))
        formal_charge = "0" if pd.isna(row.get("formal_charge")) else str(row.get("formal_charge"))

        charge = _format_float_mdf(row.get("charge"), 4)

        # Integer-like flags defaulting to 0
        def _to_int(val: Any, default: int = 0) -> int:
            try:
                if pd.isna(val):
                    return default
                return int(val)
            except Exception:
                return default

        switching_atom = _to_int(row.get("switching_atom"), 0)
        oop_flag = _to_int(row.get("oop_flag"), 0)
        chirality_flag = _to_int(row.get("chirality_flag"), 0)

        occupancy = row.get("occupancy")
        xray_temp_factor = row.get("xray_temp_factor")
        occ_val = 1.0 if pd.isna(occupancy) else occupancy
        xrf_val = 0.0 if pd.isna(xray_temp_factor) else xray_temp_factor
        occ_str = _format_float_mdf(occ_val, 4)
        xrf_str = _format_float_mdf(xrf_val, 4)

        # Preserve formal_charge token without truncation; align to width 3 only for short tokens
        fc_field = formal_charge if len(formal_charge) > 3 else f"{formal_charge:>3s}"

        conns = _compose_connections_for_atom(row, aid_to_info, adj_list, write_normalized_connections)
        line = f"{prefix:<18s} {element:>2s} {atom_type:<6s} {charge_group:<6s} {isotope:>5s} {fc_field} {charge:>8s} {switching_atom:d} {oop_flag:d} {chirality_flag:d} {occ_str:>7s}  {xrf_str:>7s}"
        if conns:
            line = f"{line} {conns}"
        lines.append(line)

    lines.append("")
    if footer_lines:
        lines.extend(footer_lines)
    else:
        lines.append("#end")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")

    return path
__all__ = ["load_mdf"]