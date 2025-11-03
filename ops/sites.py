from __future__ import annotations

"""
Site discovery utilities for grafting workflows.

This module provides general-purpose helpers to:
- Split a slab into "top" vs "bottom" surfaces by a z-threshold
- Pair two atom-type populations by distance (e.g., OH: O with H)
- Build structured site records from detected pairs
- Select sites per surface (e.g., half coverage), deterministically with a seed

These utilities operate on pandas DataFrames compatible with USM.atoms schema:
required columns: ["aid", "atom_type", "x", "y", "z"]
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class Site:
    """
    One grafting site anchored at an "O" position (or any type_a), optionally paired with H.

    Fields
    ------
    site_id: int
        Unique sequential id within the discovered sites list.
    o_aid: int
        Atom id for the O (or type_a) anchor.
    h_aid: Optional[int]
        Paired H (or type_b) atom id; may be None if unpaired or filtered.
    side: str
        "top" or "bottom" assigned by comparing O.z to the split threshold.
    o_pos: Tuple[float, float, float]
        Anchor coordinate (x, y, z).
    distance: Optional[float]
        Pair distance between O and H when applicable.
    removal_aids: List[int]
        Host atom ids to remove when consuming this site (typically [o_aid, h_aid] if h_aid exists).
    meta: Dict[str, Any]
        Free-form metadata (e.g., {"o_type": "omx", "h_type": "hoy"}).
    """
    site_id: int
    o_aid: int
    h_aid: Optional[int]
    side: str
    o_pos: Tuple[float, float, float]
    distance: Optional[float]
    removal_aids: List[int]
    meta: Dict[str, Any]


# ---------------------------
# Threshold / surface split
# ---------------------------

def compute_surface_split_threshold(z_values: Iterable[float],
                                    method: str = "auto",
                                    round_precision: int = 6) -> float:
    """
    Compute a z threshold to split top vs bottom surfaces.

    Strategies
    ----------
    - auto: round z to 'round_precision' decimals, then mid-point of min/max unique rounded values
            falling back to the median when only one rounded z exists
    - median: median(z)
    - midrange: (min(z) + max(z)) / 2

    Returns 0.0 when no finite z values are available.
    """
    z = np.asarray(list(z_values), dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0
    m = (method or "auto").lower().strip()
    if m == "auto":
        rounded = np.round(z, int(round_precision))
        uniq = np.unique(rounded)
        if uniq.size >= 2:
            return float((np.min(uniq) + np.max(uniq)) / 2.0)
        return float(np.median(z))
    if m == "median":
        return float(np.median(z))
    if m == "midrange":
        return float((float(np.min(z)) + float(np.max(z))) / 2.0)
    # default
    return float(np.median(z))


# ---------------------------
# Pairing utilities
# ---------------------------

def _validate_atoms_table(atoms: pd.DataFrame) -> None:
    req = {"aid", "atom_type", "x", "y", "z"}
    missing = [c for c in req if c not in atoms.columns]
    if missing:
        raise ValueError(f"atoms table missing required columns: {missing}")


def pair_by_distance(atoms: pd.DataFrame,
                     type_a: str,
                     type_b: str,
                     cutoff: float) -> List[Tuple[int, int, float]]:
    """
    Greedy one-to-one pairing between atoms of type_a and type_b by Euclidean distance within cutoff.

    Inputs
    ------
    atoms: DataFrame with columns ["aid","atom_type","x","y","z"]
    type_a: anchor type (e.g., "omx")
    type_b: partner type (e.g., "hoy")
    cutoff: maximum allowed distance for pairing (Angstrom)

    Returns
    -------
    List of tuples (aid_a, aid_b, distance), sorted by ascending distance, guaranteed one-to-one.
    """
    _validate_atoms_table(atoms)
    a = atoms.copy()
    a["__type"] = a["atom_type"].astype("string").str.lower()
    A = a[a["__type"] == str(type_a).lower()].copy()
    B = a[a["__type"] == str(type_b).lower()].copy()

    if A.empty or B.empty:
        return []

    A_xyz = A[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    B_xyz = B[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    A_ids = A["aid"].astype(int).to_numpy(copy=True)
    B_ids = B["aid"].astype(int).to_numpy(copy=True)

    # Pairwise distances (|A| x |B|)
    diff = A_xyz[:, None, :] - B_xyz[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d = np.sqrt(d2)

    # Candidates within cutoff
    ai, bi = np.where(d <= float(cutoff))
    if ai.size == 0:
        return []

    candidates: List[Tuple[float, int, int]] = [
        (float(d[i, j]), int(i), int(j)) for i, j in zip(ai.tolist(), bi.tolist())
    ]
    candidates.sort(key=lambda t: t[0])  # ascending by distance

    used_A: set[int] = set()
    used_B: set[int] = set()
    pairs: List[Tuple[int, int, float]] = []

    for dist, ia, ib in candidates:
        if ia in used_A or ib in used_B:
            continue
        used_A.add(ia)
        used_B.add(ib)
        pairs.append((int(A_ids[ia]), int(B_ids[ib]), float(dist)))

    return pairs


# ---------------------------
# Side assignment and site building
# ---------------------------

def classify_side_for_oids(atoms: pd.DataFrame,
                           o_aids: Iterable[int],
                           threshold_z: float) -> Dict[int, str]:
    """
    Return a mapping o_aid -> "top"/"bottom" based on z > threshold_z.
    """
    _validate_atoms_table(atoms)
    oset = set(int(x) for x in o_aids)
    rows = atoms[atoms["aid"].isin(oset)][["aid", "z"]].copy()
    if rows.empty:
        return {}
    zmap = dict(zip(rows["aid"].astype(int).tolist(), rows["z"].astype(float).tolist()))
    return {oid: ("top" if float(zmap.get(int(oid), 0.0)) > float(threshold_z) else "bottom") for oid in oset}


def build_sites_from_pairs(atoms: pd.DataFrame,
                           pairs: List[Tuple[int, int, float]],
                           threshold_z: float,
                           o_type: Optional[str] = None,
                           h_type: Optional[str] = None) -> List[Site]:
    """
    Construct Site records from (o_aid, h_aid, distance) tuples.

    o_type / h_type are stored in Site.meta for reference; not used for logic here.
    """
    _validate_atoms_table(atoms)
    if not pairs:
        return []

    # Build lookup for O positions
    aids_needed = set(int(o) for (o, _, _) in pairs)
    needed_rows = atoms[atoms["aid"].isin(aids_needed)][["aid", "x", "y", "z"]].copy()
    if needed_rows.empty:
        return []

    pos = {
        int(r["aid"]): (float(r["x"]), float(r["y"]), float(r["z"]))
        for _, r in needed_rows.iterrows()
    }
    side_map = classify_side_for_oids(atoms, [o for (o, _, _) in pairs], threshold_z)

    sites: List[Site] = []
    for sid, (oa, hb, dist) in enumerate(pairs):
        oa = int(oa)
        hb = int(hb) if hb is not None else None
        removal = [oa] + ([hb] if hb is not None else [])
        s = Site(
            site_id=int(sid),
            o_aid=oa,
            h_aid=hb,
            side=str(side_map.get(oa, "bottom")),
            o_pos=tuple(pos.get(oa, (0.0, 0.0, 0.0))),
            distance=float(dist) if dist is not None else None,
            removal_aids=removal,
            meta={"o_type": o_type, "h_type": h_type} if (o_type or h_type) else {},
        )
        sites.append(s)

    return sites


# ---------------------------
# Site selection helpers
# ---------------------------

def count_sites_by_side(sites: List[Site]) -> Dict[str, int]:
    """
    Count sites per side.
    """
    top = sum(1 for s in sites if s.side == "top")
    bot = sum(1 for s in sites if s.side == "bottom")
    return {"top": int(top), "bottom": int(bot), "total": int(len(sites))}


def pick_half_per_surface(sites: List[Site], seed: int = 42) -> List[Site]:
    """
    Pick floor(N/2) sites from each surface independently, using a deterministic RNG.
    """
    rng = np.random.default_rng(int(seed))
    top_sites = [s for s in sites if s.side == "top"]
    bot_sites = [s for s in sites if s.side == "bottom"]

    def _pick_half(lst: List[Site]) -> List[Site]:
        if not lst:
            return []
        n = len(lst) // 2
        if n <= 0:
            return []
        idx = rng.choice(len(lst), size=n, replace=False)
        return [lst[i] for i in sorted(idx.tolist())]

    return _pick_half(top_sites) + _pick_half(bot_sites)


def pick_fraction_per_surface(sites: List[Site], fraction: float, seed: int = 42) -> List[Site]:
    """
    Pick floor(fraction * N_side) sites from each surface independently.
    fraction in [0,1].
    """
    fraction = float(fraction)
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0, 1]")
    rng = np.random.default_rng(int(seed))
    top_sites = [s for s in sites if s.side == "top"]
    bot_sites = [s for s in sites if s.side == "bottom"]

    def _pick_frac(lst: List[Site]) -> List[Site]:
        if not lst:
            return []
        n = int(np.floor(fraction * len(lst)))
        if n <= 0:
            return []
        idx = rng.choice(len(lst), size=n, replace=False)
        return [lst[i] for i in sorted(idx.tolist())]

    return _pick_frac(top_sites) + _pick_frac(bot_sites)


def pick_n_per_surface(sites: List[Site], n_top: int, n_bottom: int, seed: int = 42) -> List[Site]:
    """
    Pick up to n_top sites from the top surface and up to n_bottom from the bottom, deterministically.

    - If requested n exceeds available sites on a side, all available sites on that side are returned.
    - Selection is uniform without replacement using a seeded RNG for reproducibility.
    - Returned list is sorted by site_id for stable downstream behavior.
    """
    rng = np.random.default_rng(int(seed))
    tops = [s for s in sites if s.side == "top"]
    bots = [s for s in sites if s.side == "bottom"]

    def _pick(lst: List[Site], n: int) -> List[Site]:
        n_req = max(0, int(n))
        if n_req <= 0 or not lst:
            return []
        k = min(n_req, len(lst))
        idx = rng.choice(len(lst), size=k, replace=False)
        return [lst[i] for i in sorted(idx.tolist())]

    chosen = _pick(tops, int(n_top)) + _pick(bots, int(n_bottom))
    return sorted(chosen, key=lambda s: s.site_id)

def pick_n_per_surface_spaced(sites: List[Site], n_top: int, n_bottom: int, min_spacing: float, seed: int = 42) -> List[Site]:
    """
    Greedy per-surface selection of up to n_top / n_bottom sites with a minimum in-plane
    spacing between chosen O-anchor positions.

    Strategy:
    - Work on top and bottom independently.
    - Greedy farthest-first with seeded start; accept candidates whose min distance to the
      already-chosen set is >= min_spacing. If the spacing constraint prevents reaching the requested
      count, fill remaining slots by choosing the farthest candidates (ignoring the spacing constraint).
    - Returned list is sorted by site_id for stable downstream behavior.
    """
    spacing = float(max(0.0, min_spacing))
    sp2 = spacing * spacing
    rng = np.random.default_rng(int(seed))

    def _xy(s: Site) -> Tuple[float, float]:
        x, y, _ = s.o_pos
        return float(x), float(y)

    def _select(lst: List[Site], k: int) -> List[Site]:
        k = max(0, int(k))
        if k <= 0 or not lst:
            return []
        # Positions
        XY = np.array([_xy(s) for s in lst], dtype=float)
        n = XY.shape[0]
        remaining = list(range(n))
        chosen_idx: List[int] = []
        # Start from a seeded random index for coverage diversity
        start = int(rng.integers(0, n))
        chosen_idx.append(start)
        remaining.remove(start)

        def _min_d2_to_chosen(i: int) -> float:
            if not chosen_idx:
                return float("inf")
            v = XY[i]
            C = XY[chosen_idx]
            d2 = np.sum((C - v[None, :]) ** 2, axis=1)
            return float(np.min(d2)) if d2.size > 0 else float("inf")

        # Phase 1: honor spacing
        while len(chosen_idx) < k and remaining:
            # Compute each candidate's min distance to chosen
            scores = [(i, _min_d2_to_chosen(i)) for i in remaining]
            # Filter by spacing threshold
            ok = [i for i, d2 in scores if d2 >= sp2]
            if not ok:
                break
            # Pick the candidate with the largest min distance (farthest-first)
            best = max(ok, key=lambda i: _min_d2_to_chosen(i))
            chosen_idx.append(best)
            remaining.remove(best)

        # Phase 2: fill remaining slots by farthest-first (ignoring spacing)
        while len(chosen_idx) < k and remaining:
            # pick the farthest by min distance
            best = max(remaining, key=lambda i: _min_d2_to_chosen(i))
            chosen_idx.append(best)
            remaining.remove(best)

        chosen = [lst[i] for i in sorted(chosen_idx)]
        return chosen[:k]

    tops = [s for s in sites if s.side == "top"]
    bots = [s for s in sites if s.side == "bottom"]
    chosen = _select(tops, n_top) + _select(bots, n_bottom)
    return sorted(chosen, key=lambda s: s.site_id)

def select_sites_by_ids(sites: List[Site], site_ids: Iterable[int]) -> List[Site]:
    """
    Filter to a subset of sites by site_id.
    """
    sidset = set(int(x) for x in site_ids)
    return [s for s in sites if s.site_id in sidset]


def sites_to_dataframe(sites: List[Site]) -> pd.DataFrame:
    """
    Convert a list of Site records into a DataFrame for manifests and reports.
    """
    if not sites:
        return pd.DataFrame(columns=["site_id", "o_aid", "h_aid", "side", "x", "y", "z", "distance"])
    rows: List[Dict[str, Any]] = []
    for s in sites:
        x, y, z = s.o_pos
        rows.append({
            "site_id": int(s.site_id),
            "o_aid": int(s.o_aid),
            "h_aid": (None if s.h_aid is None else int(s.h_aid)),
            "side": s.side,
            "x": float(x), "y": float(y), "z": float(z),
            "distance": (None if s.distance is None else float(s.distance)),
        })
    return pd.DataFrame(rows).sort_values(by=["site_id"]).reset_index(drop=True)


__all__ = [
    "Site",
    "compute_surface_split_threshold",
    "pair_by_distance",
    "classify_side_for_oids",
    "build_sites_from_pairs",
    "count_sites_by_side",
    "pick_half_per_surface",
    "pick_fraction_per_surface",
    "pick_n_per_surface",
    "select_sites_by_ids",
    "sites_to_dataframe",
]