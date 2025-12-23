from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from usm.ops.parameterset import ParameterSetDerivationError, derive_parameterset_v0_1_2


def test_derive_parameterset_v0_1_2_missing_required_columns_deterministic() -> None:
    # Use a duck-typed structure (not USM) so the required cols are truly missing.
    atoms = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "a"},
            {"aid": 1, "atom_type": "b"},
        ]
    )
    s = SimpleNamespace(atoms=atoms)

    with pytest.raises(ParameterSetDerivationError) as ei:
        derive_parameterset_v0_1_2(s)

    e = ei.value
    assert e.missing_types == ()
    assert e.inconsistent_types == ()
    assert e.details["missing_columns"] == [
        "lj_epsilon_kcal_mol",
        "lj_sigma_angstrom",
        "mass_amu",
    ]


def test_derive_parameterset_v0_1_2_missing_values_per_type_sorted() -> None:
    atoms = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "b", "mass_amu": 10.0, "lj_sigma_angstrom": 3.0, "lj_epsilon_kcal_mol": 0.1},
            {"aid": 1, "atom_type": "a", "mass_amu": None, "lj_sigma_angstrom": 2.0, "lj_epsilon_kcal_mol": 0.2},
            {"aid": 2, "atom_type": "a", "mass_amu": None, "lj_sigma_angstrom": 2.0, "lj_epsilon_kcal_mol": 0.2},
        ]
    )
    s = SimpleNamespace(atoms=atoms)

    with pytest.raises(ParameterSetDerivationError) as ei:
        derive_parameterset_v0_1_2(s)

    e = ei.value
    assert e.missing_types == ("a",)
    assert e.inconsistent_types == ()
    assert e.details["missing"]["a"]["columns"] == ["mass_amu"]


def test_derive_parameterset_v0_1_2_inconsistent_values_per_type_sorted() -> None:
    atoms = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "b", "mass_amu": 10.0, "lj_sigma_angstrom": 3.0, "lj_epsilon_kcal_mol": 0.1},
            {"aid": 1, "atom_type": "a", "mass_amu": 1.0, "lj_sigma_angstrom": 2.0, "lj_epsilon_kcal_mol": 0.2},
            {"aid": 2, "atom_type": "a", "mass_amu": 2.0, "lj_sigma_angstrom": 2.0, "lj_epsilon_kcal_mol": 0.2},
        ]
    )
    s = SimpleNamespace(atoms=atoms)

    with pytest.raises(ParameterSetDerivationError) as ei:
        derive_parameterset_v0_1_2(s)

    e = ei.value
    assert e.missing_types == ()
    assert e.inconsistent_types == ("a",)
    assert e.details["inconsistent"]["a"]["columns"] == ["mass_amu"]


def test_derive_parameterset_v0_1_2_success_deterministic_under_atom_row_order() -> None:
    atoms_a = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "b", "element": "B", "mass_amu": 10.0, "lj_sigma_angstrom": 3.0, "lj_epsilon_kcal_mol": 0.1},
            {"aid": 1, "atom_type": "a", "element": "A", "mass_amu": 1.0, "lj_sigma_angstrom": 2.0, "lj_epsilon_kcal_mol": 0.2},
            {"aid": 2, "atom_type": "a", "element": "A", "mass_amu": 1.0, "lj_sigma_angstrom": 2.0, "lj_epsilon_kcal_mol": 0.2},
        ]
    )
    atoms_b = atoms_a.iloc[[2, 0, 1]].reset_index(drop=True)

    ps_a = derive_parameterset_v0_1_2(SimpleNamespace(atoms=atoms_a))
    ps_b = derive_parameterset_v0_1_2(SimpleNamespace(atoms=atoms_b))
    assert ps_a == ps_b

    # Atom types keys are sorted.
    assert list(ps_a["atom_types"].keys()) == ["a", "b"]

