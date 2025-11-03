from __future__ import annotations

from typing import Iterable, Tuple, Union, Optional
import numpy as np
import pandas as pd

from usm.core.model import USM


ArrayLike = Union[Iterable[float], np.ndarray]


def _as_vec3(v: ArrayLike) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr.item(), 3)
    if arr.size != 3:
        raise ValueError("Expected a 3-vector or scalar")
    return arr.astype(float)


def _copy_or_inplace(usm: USM, in_place: bool) -> USM:
    return usm if in_place else usm.copy()


def translate(usm: USM, delta: ArrayLike, in_place: bool = False) -> USM:
    """
    Translate all atom coordinates by a constant vector.
    """
    out = _copy_or_inplace(usm, in_place)
    d = _as_vec3(delta)
    xyz = out.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    xyz = xyz + d[None, :]
    out.atoms.loc[:, ["x", "y", "z"]] = xyz
    return out


def rotation_matrix_from_axis_angle(axis: ArrayLike, angle_deg: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix using Rodrigues' formula.
    axis: rotation axis (3-vector)
    angle_deg: angle in degrees
    """
    axis = _as_vec3(axis)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("Rotation axis must be non-zero")
    u = axis / norm
    ux, uy, uz = u
    theta = np.deg2rad(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    R = np.array([
        [c + ux*ux*C,     ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s,  c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s,  uz*uy*C + ux*s, c + uz*uz*C   ],
    ], dtype=float)
    return R


def rotate(usm: USM, R: np.ndarray, origin: ArrayLike = (0.0, 0.0, 0.0), in_place: bool = False) -> USM:
    """
    Apply a 3x3 rotation matrix R to all coordinates about a specified origin.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    out = _copy_or_inplace(usm, in_place)
    o = _as_vec3(origin)
    xyz = out.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    xyz = (xyz - o[None, :]) @ R.T + o[None, :]
    out.atoms.loc[:, ["x", "y", "z"]] = xyz
    return out


def scale(usm: USM, factors: ArrayLike, origin: ArrayLike = (0.0, 0.0, 0.0), in_place: bool = False) -> USM:
    """
    Scale coordinates by scalar or per-axis factors about a specified origin.
    factors: scalar or 3-vector
    """
    out = _copy_or_inplace(usm, in_place)
    s = _as_vec3(factors)
    o = _as_vec3(origin)
    xyz = out.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    xyz = (xyz - o[None, :]) * s[None, :] + o[None, :]
    out.atoms.loc[:, ["x", "y", "z"]] = xyz
    return out


def wrap_to_cell(usm: USM, in_place: bool = False) -> USM:
    """
    Wrap coordinates back into the unit cell (orthorhombic assumption for v0.1).
    If pbc is False or cell parameters are not finite, no-op (returns copy unless in_place=True).
    """
    out = _copy_or_inplace(usm, in_place)
    cell = out.cell or {}
    if not bool(cell.get("pbc", False)):
        return out

    a = cell.get("a", np.nan)
    b = cell.get("b", np.nan)
    c = cell.get("c", np.nan)
    alpha = cell.get("alpha", 90.0)
    beta = cell.get("beta", 90.0)
    gamma = cell.get("gamma", 90.0)

    if not np.all(np.isfinite([a, b, c, alpha, beta, gamma])):
        return out

    # v0.1: assume orthorhombic (angles ~ 90 deg)
    if not (abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6):
        # Non-orthorhombic wrapping could be added later (fractional conversion), skip for now
        return out

    xyz = out.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    # Use modulo operation to bring within [0, L)
    L = np.array([a, b, c], dtype=float)
    # Avoid division by zero
    L[L == 0.0] = np.nan
    frac = xyz / L[None, :]
    frac = frac - np.floor(frac)  # fractional in [0,1)
    wrapped = frac * L[None, :]
    # Where L is NaN (undefined), keep original
    sel = np.isfinite(L)
    xyz[:, sel] = wrapped[:, sel]
    out.atoms.loc[:, ["x", "y", "z"]] = xyz
    return out