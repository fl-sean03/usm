from __future__ import annotations

"""
Collision and proximity utilities.

This module provides vectorized minimum-distance and clash checks between two
sets of atoms, with optional SciPy KD-tree acceleration when available.
"""

from typing import Iterable, Optional, Tuple
import numpy as np

from usm.core.model import USM


# Try optional SciPy acceleration
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _as_xyz(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Expected array of shape (N, 3)")
    return a


def extract_xyz(usm: USM, mask: Optional[Iterable[bool]] = None) -> np.ndarray:
    """
    Extract XYZ coordinates from a USM, optionally applying a boolean mask.

    Returns
    -------
    xyz : (N,3) ndarray
    """
    coords = usm.atoms[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    if mask is not None:
        m = np.asarray(list(mask), dtype=bool)
        if m.shape[0] != coords.shape[0]:
            raise ValueError("Mask length does not match number of atoms")
        coords = coords[m]
    return coords


def min_distance_between_sets(A_xyz: np.ndarray, B_xyz: np.ndarray, chunk: int = 4096) -> float:
    """
    Compute minimal Euclidean distance between two point sets.

    Uses SciPy KD-tree if available; otherwise falls back to chunked brute-force.

    Returns +inf if either set is empty.
    """
    A = _as_xyz(A_xyz)
    B = _as_xyz(B_xyz)
    if A.size == 0 or B.size == 0:
        return float(np.inf)

    if _HAVE_SCIPY:
        tree = cKDTree(B)  # type: ignore
        dists, _ = tree.query(A, k=1, workers=-1)
        return float(np.min(dists)) if dists.size > 0 else float(np.inf)

    # Chunked brute-force to bound memory
    best = float(np.inf)
    n = A.shape[0]
    for i0 in range(0, n, int(chunk)):
        i1 = min(n, i0 + int(chunk))
        D = A[i0:i1, None, :] - B[None, :, :]
        d2 = np.sum(D * D, axis=2)
        local = float(np.sqrt(np.min(d2))) if d2.size > 0 else float(np.inf)
        if local < best:
            best = local
            if best <= 0.0:
                return 0.0
    return best


def has_clash_between_sets(A_xyz: np.ndarray, B_xyz: np.ndarray, cutoff: float, chunk: int = 4096) -> bool:
    """
    Return True if any inter-set pair is closer than 'cutoff'.
    """
    A = _as_xyz(A_xyz)
    B = _as_xyz(B_xyz)
    if A.size == 0 or B.size == 0:
        return False

    if _HAVE_SCIPY:
        tree = cKDTree(B)  # type: ignore
        # Query within cutoff; early exit if any found
        idxs = tree.query_ball_point(A, r=float(cutoff), workers=-1)
        return any(len(lst) > 0 for lst in idxs)

    # Chunked brute-force
    c2 = float(cutoff) ** 2
    n = A.shape[0]
    for i0 in range(0, n, int(chunk)):
        i1 = min(n, i0 + int(chunk))
        D = A[i0:i1, None, :] - B[None, :, :]
        d2 = np.sum(D * D, axis=2)
        if np.any(d2 < c2):
            return True
    return False


def min_distance_guest_host(guest: USM, host: USM,
                            host_keep_mask: Optional[Iterable[bool]] = None) -> float:
    """
    Convenience wrapper operating on USM objects; allows masking the host.
    """
    gx = extract_xyz(guest)
    hx = extract_xyz(host, mask=host_keep_mask)
    return min_distance_between_sets(gx, hx)


def has_clash_guest_host(guest: USM, host: USM,
                         cutoff: float,
                         host_keep_mask: Optional[Iterable[bool]] = None) -> bool:
    """
    Return True if guest clashes with host under the given cutoff.
    """
    gx = extract_xyz(guest)
    hx = extract_xyz(host, mask=host_keep_mask)
    return has_clash_between_sets(gx, hx, cutoff)


__all__ = [
    "extract_xyz",
    "min_distance_between_sets",
    "has_clash_between_sets",
    "min_distance_guest_host",
    "has_clash_guest_host",
]