from __future__ import annotations

from dataclasses import asdict

import numpy as np

from usm.core.model import USM  # [USM()](src/usm/core/model.py:95)
from usm.ops.normalize import normalize_coords  # [normalize_coords()](src/usm/ops/normalize.py:65)


def _finite_xyz(usm: USM) -> np.ndarray:
    xyz = usm.atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def test_shift_only_makes_mins_zero_and_nonnegative() -> None:
    atoms = [
        {"x": -2.5, "y": 1.0, "z": 3.0},
        {"x": 0.0, "y": -7.0, "z": 2.0},
        {"x": 1.5, "y": 2.0, "z": -1.0},
    ]
    u0 = USM.from_records(atoms)

    u1, rep = normalize_coords(u0, mode="shift", return_report=True)
    xyz = _finite_xyz(u1)

    mins = np.min(xyz, axis=0)
    assert np.allclose(mins, 0.0)
    assert np.all(xyz >= -1e-12)
    assert rep.mode_applied == "shift"


def test_shift_deterministic_under_row_reorder() -> None:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-10.0, 10.0, size=(50, 3)).astype(np.float64)
    atoms = [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in xyz]
    u0 = USM.from_records(atoms)

    perm = rng.permutation(len(atoms))
    atoms_perm = [atoms[int(i)] for i in perm]
    u0p = USM.from_records(atoms_perm)

    u1, rep1 = normalize_coords(u0, mode="shift", return_report=True)
    u2, rep2 = normalize_coords(u0p, mode="shift", return_report=True)

    assert rep1.delta_xyz == rep2.delta_xyz

    # Compare coordinate multisets (order-independent)
    a = np.sort(_finite_xyz(u1), axis=0)
    b = np.sort(_finite_xyz(u2), axis=0)
    assert np.allclose(a, b)


def test_nan_rows_ignored_in_mins_and_preserved() -> None:
    atoms = [
        {"x": -2.0, "y": -3.0, "z": 5.0},
        {"x": float("nan"), "y": 999.0, "z": 999.0},
        {"x": 4.0, "y": -1.0, "z": -2.0},
    ]
    u0 = USM.from_records(atoms)
    u1 = normalize_coords(u0, mode="shift")

    xyz1 = u1.atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
    assert np.isnan(xyz1[1, 0])

    finite = _finite_xyz(u1)
    mins = np.min(finite, axis=0)
    assert np.allclose(mins, 0.0)


def test_wrap_shift_orthorhombic_in_bounds_and_mins_zero() -> None:
    cell = dict(pbc=True, a=10.0, b=12.0, c=8.0, alpha=90.0, beta=90.0, gamma=90.0, spacegroup="")
    atoms = [
        {"x": -1.0, "y": 0.0, "z": 0.0},
        {"x": 10.5, "y": 12.25, "z": -0.1},
        {"x": 25.0, "y": -19.0, "z": 100.0},
    ]
    u0 = USM.from_records(atoms, cell=cell)
    u1, rep = normalize_coords(u0, mode="wrap_shift", return_report=True)
    xyz = _finite_xyz(u1)

    assert rep.mode_applied == "wrap_shift"
    assert rep.pbc_used is True

    # In [0,L) within tolerance
    assert np.all(xyz[:, 0] >= -1e-12) and np.all(xyz[:, 0] < cell["a"] + 1e-12)
    assert np.all(xyz[:, 1] >= -1e-12) and np.all(xyz[:, 1] < cell["b"] + 1e-12)
    assert np.all(xyz[:, 2] >= -1e-12) and np.all(xyz[:, 2] < cell["c"] + 1e-12)
    assert np.allclose(np.min(xyz, axis=0), 0.0)


def test_resize_wrap_shift_grows_cell_to_span_plus_padding() -> None:
    # Small starting cell; coordinates span much larger
    cell = dict(pbc=True, a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0, spacegroup="")
    atoms = [
        {"x": -10.0, "y": 1.0, "z": 2.0},
        {"x": 30.0, "y": -4.0, "z": 10.0},
        {"x": 12.0, "y": 7.0, "z": -8.0},
    ]
    u0 = USM.from_records(atoms, cell=cell)
    u1, rep = normalize_coords(u0, mode="resize_wrap_shift", pad_abs=1.5, pad_frac=0.0, return_report=True)

    assert rep.pbc_used is True
    assert rep.mode_applied == "resize_wrap_shift"

    mins0, maxs0 = rep.mins_before, rep.maxs_before
    span = np.array([maxs0[i] - mins0[i] for i in range(3)], dtype=float)

    # auto_resize_cell_auto uses pad_extra = max(2*pad_abs, pad_frac*span)
    desired = span + np.maximum(2.0 * 1.5, 0.0 * span)
    assert float(u1.cell["a"]) >= float(desired[0]) - 1e-8
    assert float(u1.cell["b"]) >= float(desired[1]) - 1e-8
    assert float(u1.cell["c"]) >= float(desired[2]) - 1e-8

    # Report must be JSON-serializable
    _ = asdict(rep)

