from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from usm.ops.termset import derive_termset_v0_1_2, write_termset_json


def _make_structure(*, atoms: pd.DataFrame, bonds: pd.DataFrame | None) -> SimpleNamespace:
    return SimpleNamespace(atoms=atoms, bonds=bonds)


def test_derive_termset_v0_1_2_deterministic_under_bond_row_order(tmp_path) -> None:
    # 0=c3 -- 1=o -- 2=h
    atoms = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "c3"},
            {"aid": 1, "atom_type": "o"},
            {"aid": 2, "atom_type": "h"},
        ]
    )

    bonds_a = pd.DataFrame([{"a1": 0, "a2": 1}, {"a1": 1, "a2": 2}])
    # Shuffled order + reversed endpoints
    bonds_b = pd.DataFrame([{"a1": 2, "a2": 1}, {"a1": 1, "a2": 0}])

    s_a = _make_structure(atoms=atoms, bonds=bonds_a)
    s_b = _make_structure(atoms=atoms, bonds=bonds_b)

    t_a = derive_termset_v0_1_2(s_a)
    t_b = derive_termset_v0_1_2(s_b)
    assert t_a == t_b

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    write_termset_json(t_a, out_a)
    write_termset_json(t_b, out_b)

    txt_a = out_a.read_text(encoding="utf-8")
    txt_b = out_b.read_text(encoding="utf-8")
    assert txt_a == txt_b
    assert txt_a.endswith("\n")
    assert json.loads(txt_a) == t_a


def test_derive_termset_v0_1_2_deterministic_under_atom_row_order_with_aid_mapping() -> None:
    # Atoms DataFrame row-order changes must not matter when aid mapping exists.
    atoms_a = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "c3"},
            {"aid": 1, "atom_type": "o"},
            {"aid": 2, "atom_type": "h"},
        ]
    )
    atoms_b = atoms_a.iloc[[2, 0, 1]].reset_index(drop=True)

    bonds = pd.DataFrame([{"a1": 0, "a2": 1}, {"a1": 1, "a2": 2}])

    t_a = derive_termset_v0_1_2(_make_structure(atoms=atoms_a, bonds=bonds))
    t_b = derive_termset_v0_1_2(_make_structure(atoms=atoms_b, bonds=bonds))
    assert t_a == t_b


def test_derive_termset_v0_1_2_canonicalization_bond_angle_dihedral_improper() -> None:
    # Build a 4-atom chain for angles+dihedrals:
    # 0=z -- 1=y -- 2=x -- 3=a
    # Dihedral types from 0-1-2-3 should canonicalize to min(fwd, rev)
    # fwd=(z,y,x,a), rev=(a,x,y,z) => choose (a,x,y,z)
    #
    # Also add an improper center 4=C connected to three peripherals {5:z, 6:a, 7:m}
    # Improper key assumes central is t2 and peripherals are sorted.
    atoms = pd.DataFrame(
        [
            {"aid": 0, "atom_type": "z"},
            {"aid": 1, "atom_type": "y"},
            {"aid": 2, "atom_type": "x"},
            {"aid": 3, "atom_type": "a"},
            {"aid": 4, "atom_type": "C"},
            {"aid": 5, "atom_type": "z"},
            {"aid": 6, "atom_type": "a"},
            {"aid": 7, "atom_type": "m"},
        ]
    )

    bonds = pd.DataFrame(
        [
            {"a1": 0, "a2": 1},
            {"a1": 1, "a2": 2},
            {"a1": 2, "a2": 3},
            # Improper star around atom 4
            {"a1": 4, "a2": 5},
            {"a1": 6, "a2": 4},
            {"a1": 7, "a2": 4},
        ]
    )

    ts = derive_termset_v0_1_2(_make_structure(atoms=atoms, bonds=bonds))

    # Bonds: endpoint types canonicalized (t1<=t2)
    assert ["a", "x"] in ts["bond_types"]
    assert ["x", "y"] in ts["bond_types"]
    assert ["y", "z"] in ts["bond_types"]

    # Angles (0-1-2 gives (x,y,z) by endpoint swap because x<z; 1-2-3 gives (a,x,y) by swap)
    assert ["x", "y", "z"] in ts["angle_types"]
    assert ["a", "x", "y"] in ts["angle_types"]

    # Dihedral canonicalization: choose lexicographic min(fwd,rev)
    assert ts["dihedral_types"] == [["a", "x", "y", "z"]]
    assert ts["counts"]["dihedral_types"] == {"a|x|y|z": 1}

    # Improper canonicalization: central is C in position 2, peripherals sorted (a,m,z)
    assert ts["improper_types"] == [["a", "C", "m", "z"]]
    assert ts["counts"]["improper_types"] == {"a|C|m|z": 1}

