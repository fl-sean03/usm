from __future__ import annotations

"""
Orthorhombic PBC utilities: auto-resize, recenter, and wrap.

Goals:
- Robustly handle adding/merging molecules by resizing the periodic box to avoid wrap-around
- Keep general and reusable for future workflows (slabs, 3D systems, merges)

Key functions:
- compute_bounds(usm): min/max per axis
- auto_resize_cell(usm, ...): resize orthorhombic cell along chosen axes with padding, optional recentering and wrap
- auto_resize_cell_auto(usm, ...): automatically decide which axes and padding to apply based on current positions and cell
"""

from typing import Iterable, Tuple, Sequence
import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.transform import wrap_to_cell


def _finite_float(x) -> float:
    try:
        f = float(x)
        if np.isfinite(f):
            return f
        return float("nan")
    except Exception:
        return float("nan")


def _get_lengths_cell(cell: dict) -> Tuple[float, float, float]:
    a = _finite_float(cell.get("a"))
    b = _finite_float(cell.get("b"))
    c = _finite_float(cell.get("c"))
    return a, b, c


def compute_bounds(usm: USM) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Return ((xmin, ymin, zmin), (xmax, ymax, zmax)) for all atoms with finite coordinates.
    """
    xyz = usm.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    mask = np.isfinite(xyz).all(axis=1)
    if not mask.any():
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    pts = xyz[mask]
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return (float(mins[0]), float(mins[1]), float(mins[2])), (float(maxs[0]), float(maxs[1]), float(maxs[2]))


def auto_resize_cell(
    usm: USM,
    adjust_axes: Sequence[str] = ("a", "b", "c"),
    pad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    recenter: str = "min",               # "min" | "center" | "keep"
    wrap: bool = False,
    keep_min_original: bool = True,      # do not shrink below existing cell lengths when finite
) -> USM:
    """
    Resize an orthorhombic periodic cell to enclose all atoms with specified padding.

    Parameters
    ----------
    adjust_axes : which lattice lengths to adjust ("a","b","c")
    pad         : per-axis padding to add beyond span (Å)
    recenter    : "min" -> shift so mins map to 0; "center" -> center atoms in new box; "keep" -> no translation
    wrap        : if True, wrap to [0,L) after translation
    keep_min_original : if True, never shrink below existing finite cell lengths

    Returns
    -------
    USM (mutated copy)
    """
    out = usm.copy()
    mins, maxs = compute_bounds(out)
    span = np.array([maxs[i] - mins[i] for i in range(3)], dtype=float)
    padv = np.array(list(pad), dtype=float).reshape(3,)

    # Current cell lengths
    a0, b0, c0 = _get_lengths_cell(out.cell)
    L0 = np.array([
        a0 if np.isfinite(a0) else 0.0,
        b0 if np.isfinite(b0) else 0.0,
        c0 if np.isfinite(c0) else 0.0
    ], dtype=float)

    # Target lengths
    L_target = span + padv
    # Clamp by existing lengths if keep_min_original and those are finite
    if keep_min_original:
        # For axes we aren't adjusting, keep original finite lengths EXACTLY;
        # only fall back to computed target when original is not finite.
        keep_mask = np.array([ax not in adjust_axes for ax in ("a", "b", "c")], dtype=bool)
        L_target[keep_mask] = np.where(
            np.isfinite(L0[keep_mask]) & (L0[keep_mask] > 0.0),
            L0[keep_mask],
            L_target[keep_mask],
        )
        # For adjusting axes, also prevent shrinking below existing finite
        adj_mask = ~keep_mask
        finite_mask = np.isfinite(L0)
        both = adj_mask & finite_mask
        L_target[both] = np.maximum(L_target[both], L0[both])
    else:
        # Not keeping minimum; for non-adjusted axes, keep existing finite else take computed span
        keep_mask = np.array([ax not in adjust_axes for ax in ("a", "b", "c")], dtype=bool)
        L_target[keep_mask] = np.where(
            np.isfinite(L0[keep_mask]) & (L0[keep_mask] > 0.0),
            L0[keep_mask],
            L_target[keep_mask],
        )

    # Ensure strictly positive lengths (avoid zeros)
    L_target = np.where(L_target > 1e-12, L_target, 1.0)

    # Translate atoms if requested
    A = out.atoms
    if recenter == "min":
        delta = np.array([-mins[0], -mins[1], -mins[2]], dtype=float)
    elif recenter == "center":
        center = np.array([(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5], dtype=float)
        delta = L_target * 0.5 - center
    else:
        delta = np.array([0.0, 0.0, 0.0], dtype=float)

    if np.linalg.norm(delta) > 0.0:
        coords = A[["x", "y", "z"]].to_numpy(dtype=float)
        coords = coords + delta[None, :]
        out.atoms.loc[:, ["x", "y", "z"]] = coords

    # Update cell
    out.cell.update({
        "pbc": True,
        "a": float(L_target[0]),
        "b": float(L_target[1]),
        "c": float(L_target[2]),
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
    })

    # Optional wrap
    if wrap:
        out = wrap_to_cell(out, in_place=False)

    return out


def auto_resize_cell_auto(
    usm: USM,
    pad_abs: float = 8.0,
    pad_frac: float = 0.05,
    recenter: str = "min",
    wrap: bool = True,
    keep_min_original: bool = True,
) -> USM:
    """
    Automatically choose which axes to resize and how much padding to add based on current coordinates and cell.

    Strategy:
    - Compute span per axis
    - Compute per-axis extra length to add beyond span: pad_extra_i = max(2*pad_abs, pad_frac * span_i)
      (2*pad_abs allows ~pad_abs clearance on both sides)
    - If a finite cell length L0_i exists and L0_i >= span_i + pad_extra_i, do not change that axis.
      Otherwise, include axis in adjust_axes and target L_i = span_i + pad_extra_i
    - If no finite cell length is present, adjust all axes.

    Returns a new USM with updated orthorhombic cell and optional wrapping.
    """
    out = usm.copy()
    mins, maxs = compute_bounds(out)
    span = np.array([maxs[i] - mins[i] for i in range(3)], dtype=float)
    span = np.where(np.isfinite(span), span, 0.0)

    a0, b0, c0 = _get_lengths_cell(out.cell)
    L0 = np.array([a0, b0, c0], dtype=float)
    finite = np.isfinite(L0) & (L0 > 0.0)

    pad_extra = np.maximum(2.0 * float(pad_abs), float(pad_frac) * span)
    desired = span + pad_extra

    # Decide which axes need adjustment
    need = ~finite | (desired > (np.where(finite, L0, 0.0) - 1e-8))
    axes = []
    if need[0]: axes.append("a")
    if need[1]: axes.append("b")
    if need[2]: axes.append("c")
    if not axes:
        # Nothing to adjust; optionally still recenter/wrap into existing cell
        return auto_resize_cell(out, adjust_axes=(), pad=(0.0, 0.0, 0.0), recenter=recenter, wrap=wrap, keep_min_original=True)

    pad_tuple = (float(pad_extra[0]), float(pad_extra[1]), float(pad_extra[2]))
    return auto_resize_cell(
        out,
        adjust_axes=tuple(axes),
        pad=pad_tuple,
        recenter=recenter,
        wrap=wrap,
        keep_min_original=keep_min_original,
    )