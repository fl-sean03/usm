from __future__ import annotations

"""
General triclinic lattice helpers.

Conventions
-----------
- We construct a 3x3 matrix A whose ROWS are the lattice vectors (a_vec, b_vec, c_vec) in Cartesian space.
  With this convention:
    - frac_to_xyz:   R = F @ A
    - xyz_to_frac:   F = R @ A^{-1}
  where F and R are (N,3) arrays of fractional and Cartesian coordinates, respectively.

- Angles:
    alpha = angle between b and c
    beta  = angle between a and c
    gamma = angle between a and b
  All angles are specified in degrees.

- Numerical stability:
  * float64 throughout
  * guard against degenerate / near-singular cells (e.g., sin(gamma) ~ 0 or det(A) ~ 0)
  * cz is clamped at 0 when round-off yields a tiny negative under sqrt

Public API
----------
- lattice_matrix(a,b,c, alpha_deg, beta_deg, gamma_deg) -> A (3x3)
- lattice_inverse(A) -> A_inv
- frac_to_xyz(A, frac) -> xyz
- xyz_to_frac(A_inv, xyz) -> frac
"""

from typing import Tuple
import numpy as np

# Tolerances
_SIN_EPS = 1e-12
_DET_EPS = 1e-12
_SQ_EPS = 1e-14


def _finite_pos_float(x, name: str) -> float:
    try:
        f = float(x)
    except Exception:
        raise ValueError(f"{name} must be a finite float")
    if not np.isfinite(f):
        raise ValueError(f"{name} must be finite")
    return f


def _as_2d_vec3(x) -> Tuple[np.ndarray, bool]:
    """
    Return (arr2d, was_1d). Ensures shape (N,3) with dtype=float64.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError("Expected a 3-vector")
        return arr.reshape(1, 3), True
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, False
    raise ValueError("Expected an array of shape (3,) or (N,3)")


def lattice_matrix(a: float, b: float, c: float,
                   alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """
    Build the triclinic lattice matrix A (3x3) with ROW vectors [a_vec; b_vec; c_vec].

    Parameters
    ----------
    a, b, c      : lattice lengths (> 0)
    alpha_deg    : angle(b, c) in degrees
    beta_deg     : angle(a, c) in degrees
    gamma_deg    : angle(a, b) in degrees

    Returns
    -------
    A : np.ndarray shape (3,3), dtype float64

    Raises
    ------
    ValueError for non-finite/invalid parameters or degenerate/near-singular cells.
    """
    a = _finite_pos_float(a, "a")
    b = _finite_pos_float(b, "b")
    c = _finite_pos_float(c, "c")
    if not (a > 0 and b > 0 and c > 0):
        raise ValueError("a, b, c must be positive")

    try:
        alpha = np.deg2rad(float(alpha_deg))
        beta = np.deg2rad(float(beta_deg))
        gamma = np.deg2rad(float(gamma_deg))
    except Exception:
        raise ValueError("alpha/beta/gamma must be finite floats (degrees)")

    if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(gamma)):
        raise ValueError("alpha/beta/gamma must be finite")

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)

    if abs(sg) < _SIN_EPS:
        raise ValueError("Degenerate cell: sin(gamma) ~ 0")

    # Row vectors for the lattice basis
    a_vec = np.array([a, 0.0, 0.0], dtype=np.float64)
    b_vec = np.array([b * cg, b * sg, 0.0], dtype=np.float64)
    # c components in the a-b plane and perpendicular
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < -_SQ_EPS:
        raise ValueError("Invalid cell parameters: negative cz^2 (not numerically realizable)")
    cz = np.sqrt(max(0.0, cz_sq))
    c_vec = np.array([cx, cy, cz], dtype=np.float64)

    A = np.array([a_vec, b_vec, c_vec], dtype=np.float64)
    # Check invertibility / conditioning
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or abs(det) < _DET_EPS * (a * b * c):
        raise ValueError("Degenerate or near-singular lattice (determinant too small)")

    return A


def lattice_inverse(A: np.ndarray) -> np.ndarray:
    """
    Invert a lattice matrix A (with ROW vectors). Returns A^{-1} with dtype float64.

    Raises ValueError if A is not 3x3 or near-singular.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (3, 3):
        raise ValueError("A must be a 3x3 matrix")
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or abs(det) < _DET_EPS:
        raise ValueError("Lattice matrix is singular or near-singular")
    A_inv = np.linalg.inv(A).astype(np.float64, copy=False)
    return A_inv


def frac_to_xyz(A: np.ndarray, frac) -> np.ndarray:
    """
    Convert fractional coordinates to Cartesian: xyz = frac @ A.
    Accepts (3,) or (N,3) frac. Returns shape matching the input (3,) or (N,3).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (3, 3):
        raise ValueError("A must be a 3x3 matrix")
    f2d, was1d = _as_2d_vec3(frac)
    xyz = f2d @ A
    return xyz.reshape(3,) if was1d else xyz


def xyz_to_frac(A_inv: np.ndarray, xyz) -> np.ndarray:
    """
    Convert Cartesian to fractional using the inverse lattice: frac = xyz @ A_inv.
    Accepts (3,) or (N,3) xyz. Returns shape matching the input (3,) or (N,3).
    """
    A_inv = np.asarray(A_inv, dtype=np.float64)
    if A_inv.shape != (3, 3):
        raise ValueError("A_inv must be a 3x3 matrix")
    r2d, was1d = _as_2d_vec3(xyz)
    frac = r2d @ A_inv
    return frac.reshape(3,) if was1d else frac


__all__ = [
    "lattice_matrix",
    "lattice_inverse",
    "frac_to_xyz",
    "xyz_to_frac",
]