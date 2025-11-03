from __future__ import annotations

"""
Frame/orientation utilities for grafting and placement.

Provides:
- unit_vector: normalize a 3-vector
- choose_perp_axis: a stable axis perpendicular to a given vector
- rotation_matrix_from_vectors: rotate v_from to v_to (handles parallel/antiparallel)
- rotate_about_point: apply a 3x3 rotation around a given origin to a USM
- rotate_about_axis_at_point: axis-angle rotation at a given origin to a USM
- translate_to_point: translate USM so 'from_point' maps to 'to_point'
- find_aid_by_name: resolve aid for a named atom
- bonded_neighbors: list neighbor aids using bonds table
- anchor_axis_from_neighbor: define an anchor axis from an anchor atom to a preferred bonded neighbor
- outward_plane_axis: pick an in-plane axis orthogonal to a normal for outward tilt
- apply_torsion_and_tilt: compose torsion-about-normal and small outward tilt about a plane axis
"""

from typing import Iterable, Optional, Sequence, List, Tuple
import numpy as np

from usm.core.model import USM
from usm.ops.transform import (
    rotate as rotate_usm,
    translate as translate_usm,
    rotation_matrix_from_axis_angle,
)


# ---------------------------
# Basic vector helpers
# ---------------------------

def _as_vec3(v: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(v), dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr.item(), 3)
    if arr.size != 3:
        raise ValueError("Expected a 3-vector or scalar")
    return arr.astype(float)


def unit_vector(v: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    u = _as_vec3(v)
    n = float(np.linalg.norm(u))
    if n < eps:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return (u / n).astype(float)


def choose_perp_axis(v: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """
    Choose a deterministic unit vector perpendicular to v.
    """
    u = unit_vector(v, eps=eps)
    # Try cross with +x; if nearly colinear, use +y
    c = np.cross(u, np.array([1.0, 0.0, 0.0], dtype=float))
    if np.linalg.norm(c) < eps:
        c = np.cross(u, np.array([0.0, 1.0, 0.0], dtype=float))
    return unit_vector(c, eps=eps)


# ---------------------------
# Rotations and transforms
# ---------------------------

def rotation_matrix_from_vectors(v_from: Iterable[float], v_to: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """
    Create a rotation matrix R such that R @ v_from == v_to (for non-degenerate cases).
    Handles parallel and antiparallel cases deterministically.

    Returns
    -------
    R : (3,3) ndarray
    """
    a = unit_vector(v_from, eps=eps)
    b = unit_vector(v_to, eps=eps)
    if np.allclose(a, 0.0, atol=eps) or np.allclose(b, 0.0, atol=eps):
        return np.eye(3, dtype=float)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if 1.0 - dot <= 1e-12:
        # Already aligned
        return np.eye(3, dtype=float)
    if dot + 1.0 <= 1e-12:
        # Antiparallel: rotate 180° about any axis perpendicular to 'a'
        axis = choose_perp_axis(a, eps=eps)
        return rotation_matrix_from_axis_angle(axis, 180.0)

    axis = np.cross(a, b)
    angle_deg = float(np.degrees(np.arccos(dot)))
    return rotation_matrix_from_axis_angle(axis, angle_deg)


def rotate_about_point(usm: USM, R: np.ndarray, origin: Iterable[float], in_place: bool = False) -> USM:
    """
    Apply a 3x3 rotation matrix about 'origin' to all coordinates.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    return rotate_usm(usm, R, origin=_as_vec3(origin), in_place=in_place)


def rotate_about_axis_at_point(usm: USM,
                               axis_dir: Iterable[float],
                               angle_deg: float,
                               origin: Iterable[float],
                               in_place: bool = False) -> USM:
    """
    Rotate about a specified axis (through 'origin') by angle in degrees.
    """
    R = rotation_matrix_from_axis_angle(axis_dir, float(angle_deg))
    return rotate_about_point(usm, R, origin, in_place=in_place)


def translate_to_point(usm: USM,
                       from_point: Iterable[float],
                       to_point: Iterable[float],
                       in_place: bool = False) -> USM:
    """
    Translate USM so that 'from_point' (in world coords) moves to 'to_point'.
    """
    p_from = _as_vec3(from_point)
    p_to = _as_vec3(to_point)
    delta = (p_to - p_from).astype(float)
    return translate_usm(usm, delta, in_place=in_place)


# ---------------------------
# Atom lookup / neighbor utilities
# ---------------------------

def find_aid_by_name(usm: USM, name: str) -> int:
    """
    Resolve the first aid for a given atom 'name'.
    """
    a = usm.atoms
    if "name" not in a.columns:
        raise KeyError("USM.atoms has no 'name' column")
    rows = a[a["name"].astype("string") == str(name)]
    if rows.empty:
        raise KeyError(f"Atom with name '{name}' not found")
    return int(rows.iloc[0]["aid"])


def bonded_neighbors(usm: USM, aid: int) -> List[int]:
    """
    Return a sorted list of neighbor aids for atom 'aid' using usm.bonds if present.
    """
    if usm.bonds is None or len(usm.bonds) == 0:
        return []
    b = usm.bonds
    sel = b[(b["a1"] == int(aid)) | (b["a2"] == int(aid))]
    if sel.empty:
        return []
    out: set[int] = set()
    for _, r in sel.iterrows():
        a1 = int(r["a1"])
        a2 = int(r["a2"])
        out.add(a2 if a1 == int(aid) else a1)
    return sorted(out)


# ---------------------------
# Anchor axis helpers
# ---------------------------

def anchor_axis_from_neighbor(guest: USM,
                              anchor_name: str,
                              prefer_elements: Sequence[str] = ("C", "N", "O"),
                              eps: float = 1e-12) -> np.ndarray:
    """
    Define an anchor axis for the guest molecule from the anchor atom toward a preferred bonded neighbor.

    Selection policy:
    - Use guest.bonds to find neighbors of 'anchor_name'
    - Among neighbors, choose the first whose element is in prefer_elements, in order
    - If none match or no bonds, fall back to nearest atom by Euclidean distance (excluding anchor)
    - Returns a unit direction vector (neighbor - anchor) or [0,0,1] if degenerate
    """
    anchor_aid = find_aid_by_name(guest, anchor_name)
    atoms = guest.atoms.copy()
    # Get anchor position
    arow = atoms[atoms["aid"] == anchor_aid]
    if arow.empty:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    pa = arow[["x", "y", "z"]].to_numpy(dtype=float).reshape(3,)

    nb = bonded_neighbors(guest, anchor_aid)
    chosen: Optional[int] = None
    if nb:
        # Map neighbor aid -> element and pick by preference
        elem_by_aid = {}
        s = atoms[atoms["aid"].isin(nb)][["aid", "element"]]
        for _, r in s.iterrows():
            elem_by_aid[int(r["aid"])] = ("" if r["element"] is None else str(r["element"]))
        for el in prefer_elements:
            cand = [nid for nid in nb if elem_by_aid.get(int(nid), "").upper() == str(el).upper()]
            if cand:
                chosen = int(sorted(cand)[0])
                break
        if chosen is None:
            chosen = int(nb[0])

    if chosen is None:
        # Fallback: nearest other atom
        others = atoms[atoms["aid"] != anchor_aid].copy()
        if others.empty:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        xyz = others[["x", "y", "z"]].to_numpy(dtype=float)
        d2 = np.sum((xyz - pa[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        pb = xyz[idx, :]
    else:
        pb = atoms[atoms["aid"] == int(chosen)][["x", "y", "z"]].to_numpy(dtype=float).reshape(3,)

    v = pb - pa
    u = unit_vector(v, eps=eps)
    if np.allclose(u, 0.0, atol=eps):
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return u


def outward_plane_axis(normal_dir: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """
    Given a surface normal, pick a stable in-plane axis orthogonal to the normal.
    This can be used as a tilt axis to displace a guest "outward" consistently.
    """
    n = unit_vector(normal_dir, eps=eps)
    # Try axis in plane via cross with x; if degenerate, cross with y
    axis = np.cross(n, np.array([1.0, 0.0, 0.0], dtype=float))
    if np.linalg.norm(axis) < eps:
        axis = np.cross(n, np.array([0.0, 1.0, 0.0], dtype=float))
    return unit_vector(axis, eps=eps)


# ---------------------------
# Orientation composition
# ---------------------------

def apply_torsion_and_tilt(guest: USM,
                           anchor_point: Iterable[float],
                           normal_dir: Iterable[float],
                           torsion_deg: float,
                           tilt_deg: float = 0.0,
                           plane_axis: Optional[Iterable[float]] = None,
                           in_place: bool = False) -> USM:
    """
    Compose torsion and tilt rotations for a guest placed at anchor_point.

    Steps:
    - Torsion: rotate about the surface normal by torsion_deg
    - Tilt: rotate about an in-plane axis orthogonal to the normal by tilt_deg

    Note:
    - Positive tilt direction is set by 'plane_axis' if provided; otherwise a stable axis
      perpendicular to the normal is chosen (outward direction convention is the caller's choice).
    """
    out = guest if in_place else guest.copy()
    n = unit_vector(normal_dir)
    if not np.allclose(n, 0.0, atol=1e-12):
        out = rotate_about_axis_at_point(out, n, float(torsion_deg), anchor_point, in_place=True)
        if abs(float(tilt_deg)) > 0.0:
            ax = unit_vector(plane_axis) if plane_axis is not None else outward_plane_axis(n)
            out = rotate_about_axis_at_point(out, ax, float(tilt_deg), anchor_point, in_place=True)
    return out


__all__ = [
    "unit_vector",
    "choose_perp_axis",
    "rotation_matrix_from_vectors",
    "rotate_about_point",
    "rotate_about_axis_at_point",
    "translate_to_point",
    "find_aid_by_name",
    "bonded_neighbors",
    "anchor_axis_from_neighbor",
    "outward_plane_axis",
    "apply_torsion_and_tilt",
]