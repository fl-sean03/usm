from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np

from usm.core.model import USM
from usm.ops.cell import compute_bounds, auto_resize_cell_auto
from usm.ops.transform import wrap_to_cell, shift_mins_to_zero


NormalizeMode = Literal["shift", "wrap_shift", "resize_wrap_shift"]


@dataclass(frozen=True)
class CoordNormalizeReport:
    """Deterministic metadata for coordinate normalization.

    This object is intentionally small and JSON-serializable via `dataclasses.asdict()`.
    """

    mode_requested: str
    mode_applied: str
    axes: Tuple[str, ...]
    finite_only: bool
    finite_rows: int
    pbc_used: bool
    delta_xyz: Tuple[float, float, float]
    mins_before: Tuple[float, float, float]
    maxs_before: Tuple[float, float, float]
    mins_after: Tuple[float, float, float]
    maxs_after: Tuple[float, float, float]
    cell_before: dict
    cell_after: dict


def _cell_pbc_valid_for_wrap(cell: dict) -> bool:
    if not bool((cell or {}).get("pbc", False)):
        return False
    try:
        vals = np.array([
            float(cell.get("a", np.nan)),
            float(cell.get("b", np.nan)),
            float(cell.get("c", np.nan)),
            float(cell.get("alpha", 90.0)),
            float(cell.get("beta", 90.0)),
            float(cell.get("gamma", 90.0)),
        ], dtype=float)
    except Exception:
        return False
    return bool(np.all(np.isfinite(vals)))


def _cell_looks_orthorhombic(cell: dict, tol_deg: float = 1e-6) -> bool:
    if not _cell_pbc_valid_for_wrap(cell):
        return False
    try:
        alpha = float(cell.get("alpha", 90.0))
        beta = float(cell.get("beta", 90.0))
        gamma = float(cell.get("gamma", 90.0))
    except Exception:
        return False
    return (abs(alpha - 90.0) < tol_deg) and (abs(beta - 90.0) < tol_deg) and (abs(gamma - 90.0) < tol_deg)


def normalize_coords(
    usm: USM,
    *,
    mode: NormalizeMode = "shift",
    axes: Sequence[str] = ("x", "y", "z"),
    pad_abs: float = 0.0,
    pad_frac: float = 0.0,
    keep_min_original: bool = True,
    finite_only: bool = True,
    in_place: bool = False,
    return_report: bool = False,
) -> USM | tuple[USM, CoordNormalizeReport]:
    """Normalize coordinates deterministically.

    Modes
    -----
    - "shift": translate so finite mins become 0 along selected axes.
    - "wrap_shift": if PBC cell is valid, wrap into the cell first, then shift mins to 0.
      If PBC/cell invalid, deterministically degrades to "shift".
    - "resize_wrap_shift": for orthorhombic valid PBC cells, (wrap -> resize -> wrap) using
      [auto_resize_cell_auto()](src/usm/ops/cell.py:151) then shift mins to 0. If not applicable,
      degrades to "wrap_shift" or "shift".

    NaNs / non-finite
    --------------
    If finite_only=True, mins/maxs reductions ignore rows with any non-finite xyz.
    """

    axes_t = tuple(str(a) for a in axes)
    mins0, maxs0 = compute_bounds(usm)
    xyz0 = usm.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    finite_rows = int(np.isfinite(xyz0).all(axis=1).sum()) if finite_only else int(len(xyz0))

    cell_before = dict(usm.cell or {})
    requested: str = str(mode)
    applied: str = requested
    pbc_used = False
    out = usm if in_place else usm.copy()

    if requested == "shift":
        out, delta = shift_mins_to_zero(out, axes=axes_t, finite_only=finite_only, in_place=True, return_delta=True)

    elif requested == "wrap_shift":
        if _cell_pbc_valid_for_wrap(out.cell):
            pbc_used = True
            out = wrap_to_cell(out, in_place=True)
            out, delta = shift_mins_to_zero(out, axes=axes_t, finite_only=finite_only, in_place=True, return_delta=True)
        else:
            applied = "shift"
            out, delta = shift_mins_to_zero(out, axes=axes_t, finite_only=finite_only, in_place=True, return_delta=True)

    elif requested == "resize_wrap_shift":
        if _cell_looks_orthorhombic(out.cell):
            pbc_used = True
            # IMPORTANT: do NOT pre-wrap here.
            # We want the resized orthorhombic cell to enclose the *current* coordinates
            # (which may be negative/out-of-box), then wrap into the resized cell.
            out = auto_resize_cell_auto(
                out,
                pad_abs=float(pad_abs),
                pad_frac=float(pad_frac),
                recenter="min",
                wrap=True,
                keep_min_original=bool(keep_min_original),
            )
            # auto_resize_cell_auto(recenter="min") should already ensure mins==0 for finite coords,
            # but keep this step for determinism/robustness when axes subset is requested.
            out, delta = shift_mins_to_zero(out, axes=axes_t, finite_only=finite_only, in_place=True, return_delta=True)
        else:
            # degrade deterministically
            applied = "wrap_shift"
            if _cell_pbc_valid_for_wrap(out.cell):
                pbc_used = True
                out = wrap_to_cell(out, in_place=True)
                out, delta = shift_mins_to_zero(out, axes=axes_t, finite_only=finite_only, in_place=True, return_delta=True)
            else:
                applied = "shift"
                out, delta = shift_mins_to_zero(out, axes=axes_t, finite_only=finite_only, in_place=True, return_delta=True)

    else:
        raise ValueError(f"Unknown mode: {requested}")

    mins1, maxs1 = compute_bounds(out)
    report = CoordNormalizeReport(
        mode_requested=requested,
        mode_applied=applied,
        axes=axes_t,
        finite_only=bool(finite_only),
        finite_rows=finite_rows,
        pbc_used=bool(pbc_used),
        delta_xyz=(float(delta[0]), float(delta[1]), float(delta[2])),
        mins_before=(float(mins0[0]), float(mins0[1]), float(mins0[2])),
        maxs_before=(float(maxs0[0]), float(maxs0[1]), float(maxs0[2])),
        mins_after=(float(mins1[0]), float(mins1[1]), float(mins1[2])),
        maxs_after=(float(maxs1[0]), float(maxs1[1]), float(maxs1[2])),
        cell_before=cell_before,
        cell_after=dict(out.cell or {}),
    )

    return (out, report) if return_report else out


__all__ = ["normalize_coords", "CoordNormalizeReport", "NormalizeMode"]
