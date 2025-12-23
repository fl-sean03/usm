from __future__ import annotations

"""
CIF parsing internals.

This module contains the low-level CIF format parsing functions:
- Tokenizer for CIF text
- Parser for CIF data items and loops
- Symmetry operation parsing
- Element inference from labels

These are internal implementation details for the public API in cif.py.
"""

import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


_NUM_WITH_ESD_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?$")


def _strip_quotes(s: str) -> str:
    """Strip CIF-style single or double quotes from a string."""
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
    """Represents a CIF loop_ structure with tags and row values."""

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


def _parse_symop_string(s: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a CIF symmetry operation string (e.g., '-x, y+1/2, -z+1/2')
    into a (3, 3) rotation matrix and a (3,) translation vector.
    """
    rot = np.zeros((3, 3), dtype=np.float64)
    trans = np.zeros(3, dtype=np.float64)
    parts = s.replace(" ", "").lower().split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid symmetry operation string: {s}")

    for i, part in enumerate(parts):
        # Handle translation (fractions like 1/2)
        m_trans = re.findall(r"([+-]?\d+/\d+)", part)
        for val in m_trans:
            num, den = val.split("/")
            trans[i] += float(num) / float(den)
            part = part.replace(val, "")

        # Handle simple decimals/integers for translation if any remain
        m_dec = re.findall(r"([+-]?\d+\.\d+|[+-]?\d+)", part)
        for val in m_dec:
            # Only count as translation if not immediately followed by x, y, or z
            # (which would be a coefficient, rare in CIF but possible in theory)
            # Actually, standard CIF symops are very regular.
            idx = part.find(val)
            if idx + len(val) < len(part) and part[idx + len(val)] in "xyz":
                continue
            trans[i] += float(val)
            part = part.replace(val, "", 1)

        # Handle rotation components
        if "x" in part:
            rot[i, 0] = -1.0 if "-x" in part else 1.0
        if "y" in part:
            rot[i, 1] = -1.0 if "-y" in part else 1.0
        if "z" in part:
            rot[i, 2] = -1.0 if "-z" in part else 1.0

    return rot, trans


def _parse_symmetry_code(code: str) -> Tuple[int, np.ndarray]:
    """
    Parse a CIF symmetry code (e.g., '3_556' or '2') into:
    - operator_index (0-based)
    - translation vector (ix, iy, iz)
    """
    if "_" in code:
        op_part, trans_part = code.split("_")
        op_idx = int(op_part) - 1
        if len(trans_part) == 3:
            ix = int(trans_part[0]) - 5
            iy = int(trans_part[1]) - 5
            iz = int(trans_part[2]) - 5
            return op_idx, np.array([ix, iy, iz], dtype=np.int32)
        else:
            # Handle non-standard translation codes if they exist, but 555 is standard
            return op_idx, np.zeros(3, dtype=np.int32)
    else:
        try:
            return int(code) - 1, np.zeros(3, dtype=np.int32)
        except ValueError:
            # Handle '.' or other non-numeric codes as identity
            return 0, np.zeros(3, dtype=np.int32)


__all__ = [
    "_CifLoop",
    "_cif_tokenize",
    "_find_atom_site_loop",
    "_infer_element_from_label",
    "_parse_cif",
    "_parse_cif_number",
    "_parse_symop_string",
    "_parse_symmetry_code",
    "_strip_quotes",
]
