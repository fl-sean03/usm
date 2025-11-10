from __future__ import annotations

from typing import Iterable, Tuple, Union, Optional
import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.lattice import lattice_matrix, lattice_inverse, xyz_to_frac, frac_to_xyz


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
    Wrap coordinates back into the unit cell for general triclinic lattices.
    Behavior:
      - If pbc is False or cell parameters are non-finite/degenerate, no-op (returns copy unless in_place=True).
      - For orthorhombic cells (α=β=γ≈90°), use a fast-path modulo per axis.
      - Otherwise, convert to fractional coordinates via lattice inverse, wrap into [0,1), and convert back.
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

    vals = np.array([a, b, c, alpha, beta, gamma], dtype=float)
    if not np.all(np.isfinite(vals)):
        return out

    xyz = out.atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)

    # Orthorhombic fast path (preserve existing behavior/performance).
    if (abs(alpha - 90.0) < 1e-6) and (abs(beta - 90.0) < 1e-6) and (abs(gamma - 90.0) < 1e-6):
        L = np.array([a, b, c], dtype=np.float64)
        # Guard against zeros (shouldn't occur if finite and valid, but keep prior behavior)
        L[L == 0.0] = np.nan
        frac = xyz / L[None, :]
        frac = frac - np.floor(frac)
        wrapped = frac * L[None, :]
        sel = np.isfinite(L)
        xyz[:, sel] = wrapped[:, sel]
        out.atoms.loc[:, ["x", "y", "z"]] = xyz
        return out

    # General triclinic path using fractional coordinates
    try:
        A = lattice_matrix(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
        A_inv = lattice_inverse(A)
    except Exception:
        # Invalid/degenerate cell: no-op
        return out

    frac = xyz_to_frac(A_inv, xyz)  # shape (N,3)
    frac_wrapped = frac - np.floor(frac)  # [0,1)
    xyz_wrapped = frac_to_xyz(A, frac_wrapped)
    out.atoms.loc[:, ["x", "y", "z"]] = xyz_wrapped.astype(np.float64, copy=False)
    return out