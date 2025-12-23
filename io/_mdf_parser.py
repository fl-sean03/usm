"""
MDF format parsing internals.

This module contains regex patterns and helper functions for parsing
Materials Studio MDF (Material Design File) format files.

Private module - not for direct import. Use `usm.io.mdf` instead.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


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


def split_sections(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
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


def current_molecule_name_from_header(header_lines: List[str]) -> Optional[str]:
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


def molecule_order(header_lines: List[str]) -> List[str]:
    """Extract the order of @molecule declarations from header lines."""
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


def parse_atom_line(line: str, default_mol_block_name: Optional[str]) -> Dict[str, Any]:
    """Parse a single MDF atom line into a dictionary of fields."""
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


def build_bonds_from_connections(atoms_df: pd.DataFrame) -> pd.DataFrame:
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
