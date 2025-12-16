from __future__ import annotations

import json

import pytest

from usm.core.model import USM
from usm.ops.requirements import derive_requirements_v0_1, write_requirements_json


def _make_usm(*, atoms_records, bonds_records=None) -> USM:
    return USM.from_records(atoms_records=atoms_records, bonds_records=bonds_records)


def test_derive_requirements_v0_1_linear_chain_canonical() -> None:
    # 0=c3 -- 1=o -- 2=h
    usm = _make_usm(
        atoms_records=[
            {
                "aid": 0,
                "name": "C1",
                "element": "C",
                "atom_type": "c3",
                "charge": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
            {
                "aid": 1,
                "name": "O1",
                "element": "O",
                "atom_type": "o",
                "charge": 0.0,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
            {
                "aid": 2,
                "name": "H1",
                "element": "H",
                "atom_type": "h",
                "charge": 0.0,
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
        ],
        bonds_records=[{"a1": 0, "a2": 1}, {"a1": 1, "a2": 2}],
    )

    req = derive_requirements_v0_1(usm)

    assert req["atom_types"] == ["c3", "h", "o"]
    # Bond endpoints are canonicalized by type (t1 <= t2) and list is stable-sorted.
    assert req["bond_types"] == [["c3", "o"], ["h", "o"]]
    # Angle endpoints are canonicalized so t1 <= t3; only one angle in a linear chain.
    assert req["angle_types"] == [["c3", "o", "h"]]
    assert req["dihedral_types"] == []


def test_derive_requirements_v0_1_angle_endpoint_canonicalization() -> None:
    # Build an angle where endpoints should be swapped by lexicographic endpoint canonicalization:
    # z -- m -- a  => (a,m,z)
    usm = _make_usm(
        atoms_records=[
            {
                "aid": 0,
                "name": "Z1",
                "element": "X",
                "atom_type": "z",
                "charge": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
            {
                "aid": 1,
                "name": "M1",
                "element": "X",
                "atom_type": "m",
                "charge": 0.0,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
            {
                "aid": 2,
                "name": "A1",
                "element": "X",
                "atom_type": "a",
                "charge": 0.0,
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
        ],
        bonds_records=[{"a1": 0, "a2": 1}, {"a1": 1, "a2": 2}],
    )

    req = derive_requirements_v0_1(usm)

    # Bond type endpoint canonicalization: (a,m) and (m,z)
    assert req["bond_types"] == [["a", "m"], ["m", "z"]]
    # Angle endpoint canonicalization: (z,m,a) => (a,m,z)
    assert req["angle_types"] == [["a", "m", "z"]]


def test_derive_requirements_v0_1_deterministic_under_bond_row_order(tmp_path) -> None:
    atoms_records = [
        {
            "aid": 0,
            "name": "C1",
            "element": "C",
            "atom_type": "c3",
            "charge": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "mol_label": "XXXX",
            "mol_index": 1,
            "mol_block_name": "M",
        },
        {
            "aid": 1,
            "name": "O1",
            "element": "O",
            "atom_type": "o",
            "charge": 0.0,
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "mol_label": "XXXX",
            "mol_index": 1,
            "mol_block_name": "M",
        },
        {
            "aid": 2,
            "name": "H1",
            "element": "H",
            "atom_type": "h",
            "charge": 0.0,
            "x": 2.0,
            "y": 0.0,
            "z": 0.0,
            "mol_label": "XXXX",
            "mol_index": 1,
            "mol_block_name": "M",
        },
    ]

    usm_a = _make_usm(atoms_records=atoms_records, bonds_records=[{"a1": 0, "a2": 1}, {"a1": 1, "a2": 2}])

    # Shuffled row order and reversed endpoints (should not affect output).
    usm_b = _make_usm(atoms_records=atoms_records, bonds_records=[{"a1": 2, "a2": 1}, {"a1": 1, "a2": 0}])

    req_a = derive_requirements_v0_1(usm_a)
    req_b = derive_requirements_v0_1(usm_b)
    assert req_a == req_b

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    write_requirements_json(usm_a, out_a)
    write_requirements_json(usm_b, out_b)

    txt_a = out_a.read_text(encoding="utf-8")
    txt_b = out_b.read_text(encoding="utf-8")
    assert txt_a == txt_b

    # Ensure output is valid JSON and newline-terminated.
    assert txt_a.endswith("\n")
    assert json.loads(txt_a) == req_a


def test_derive_requirements_v0_1_no_bonds_graceful() -> None:
    usm = _make_usm(
        atoms_records=[
            {
                "aid": 0,
                "name": "C1",
                "element": "C",
                "atom_type": "c3",
                "charge": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            }
        ],
        bonds_records=None,
    )

    req = derive_requirements_v0_1(usm)
    assert req["atom_types"] == ["c3"]
    assert req["bond_types"] == []
    assert req["angle_types"] == []
    assert req["dihedral_types"] == []


@pytest.mark.parametrize(
    "bonds_records",
    [
        [{"a1": None, "a2": 0}],
        [{"a1": 0, "a2": None}],
        [{"a1": 0, "a2": 2}],  # out of range for 2-atom system
        [{"a1": 1, "a2": 1}],  # self-bond
    ],
)
def test_derive_requirements_v0_1_invalid_bonds_hard_error(bonds_records) -> None:
    usm = _make_usm(
        atoms_records=[
            {
                "aid": 0,
                "name": "A1",
                "element": "X",
                "atom_type": "a",
                "charge": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
            {
                "aid": 1,
                "name": "B1",
                "element": "X",
                "atom_type": "b",
                "charge": 0.0,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "mol_label": "XXXX",
                "mol_index": 1,
                "mol_block_name": "M",
            },
        ],
        bonds_records=bonds_records,
    )

    with pytest.raises(ValueError):
        derive_requirements_v0_1(usm)