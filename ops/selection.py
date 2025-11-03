from __future__ import annotations

"""
Shared selection and pairing helpers for MXN and similar workspaces.

Provides:
- split_threshold(z, method='auto'|'median'|'midrange', round_precision=6) -> float
- pair_oh_by_distance(o_df, h_df, cutoff) -> List[(o_aid, h_aid, distance)]
- count_by_side(atoms, type_list, thr) -> Dict[type, {top,bottom,total}]
"""

from typing import Iterable, List, Tuple, Dict
import numpy as np
import pandas as pd


def split_threshold(z: np.ndarray | Iterable[float], method: str = "auto", round_precision: int = 6) -> float:
    """
    Compute a z-split threshold.
    - auto: midrange of unique rounded z values if 2+ groups; else median
    - median: median(z)
    - midrange: (min(z)+max(z))/2
    """
    arr = np.asarray(z, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    m = method.lower().strip()
    if m == "auto":
        rounded = np.round(arr, int(round_precision))
        uniq = np.unique(rounded)
        if uniq.size >= 2:
            return float((np.min(uniq) + np.max(uniq)) / 2.0)
        return float(np.median(arr))
    if m == "median":
        return float(np.median(arr))
    if m == "midrange":
        return float((float(np.min(arr)) + float(np.max(arr))) / 2.0)
    return float(np.median(arr))


def pair_oh_by_distance(o_df: pd.DataFrame, h_df: pd.DataFrame, cutoff: float) -> List[Tuple[int, int, float]]:
    """
    Build O-H pairs by nearest-neighbor within cutoff Angstroms.
    Returns a one-to-one greedy assignment sorted by ascending distance:
      List[(o_aid, h_aid, distance)]
    Requires columns: 'aid','x','y','z' in each DataFrame.
    """
    if o_df is None or h_df is None or o_df.empty or h_df.empty:
        return []
    O_xyz = o_df[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    H_xyz = h_df[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    O_aids = o_df["aid"].astype(int).to_numpy(copy=True)
    H_aids = h_df["aid"].astype(int).to_numpy(copy=True)

    diff = O_xyz[:, None, :] - H_xyz[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    d = np.sqrt(d2)

    cand_o_idx, cand_h_idx = np.where(d <= float(cutoff))
    if cand_o_idx.size == 0:
        return []
    cands: List[Tuple[float, int, int]] = []
    for oi, hi in zip(cand_o_idx.tolist(), cand_h_idx.tolist()):
        cands.append((float(d[oi, hi]), int(oi), int(hi)))
    cands.sort(key=lambda t: t[0])

    used_o: set[int] = set()
    used_h: set[int] = set()
    pairs: List[Tuple[int, int, float]] = []
    for dist, oi, hi in cands:
        if oi in used_o or hi in used_h:
            continue
        used_o.add(oi)
        used_h.add(hi)
        pairs.append((int(O_aids[oi]), int(H_aids[hi]), float(dist)))
    return pairs


def count_by_side(atoms: pd.DataFrame, type_list: List[str], thr: float) -> Dict[str, Dict[str, int]]:
    """
    Count top/bottom/total for each requested atom_type relative to threshold thr.
    Expects columns: 'atom_type' and 'z'.
    """
    out: Dict[str, Dict[str, int]] = {}
    if atoms is None or len(atoms) == 0:
        return {str(t).lower(): {"top": 0, "bottom": 0, "total": 0} for t in type_list}
    a = atoms.copy()
    a["__type"] = a["atom_type"].astype("string").str.lower()
    a["__top"] = a["z"].astype(float) > float(thr)
    for t in type_list:
        tl = str(t).lower()
        rows = a[a["__type"] == tl]
        out[tl] = {
            "top": int(rows["__top"].sum()),
            "bottom": int((~rows["__top"]).sum()),
            "total": int(len(rows)),
        }
    return out


__all__ = ["split_threshold", "pair_oh_by_distance", "count_by_side"]