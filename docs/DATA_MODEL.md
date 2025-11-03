Unified Structure Model (USM) v0.1 — Data Model

Scope
USM provides a unified, columnar data model for structures parsed from Materials Studio CAR and MDF. CAR and MDF are normalized into the same in-memory schema, enabling composition (e.g., CAR coords + MDF topology) and round-trip writing with preserved headers/footers.

Core tables and metadata
- Atoms table (required)
  - Columns (name: dtype; units)
    - aid: Int32; row id (dense 0..N-1)
    - name: string
    - element: string
    - atom_type: string
    - charge: float32; e
    - x: float64; Å
    - y: float64; Å
    - z: float64; Å
    - mol_label: string (e.g., MOL2 or XXXX from file prefixes)
    - mol_index: Int32 (e.g., 1 from MOL2_1 or XXXX 1)
    - mol_block_name: string (MDF @molecule name; empty for CAR-only)
    - MDF carry-through (nullable): isotope (string), formal_charge (string), switching_atom (Int8), oop_flag (Int8), chirality_flag (Int8), occupancy (float32), xray_temp_factor (float32), charge_group (string), connections_raw (string; exact MDF tokens)
  - Required columns are enforced in [REQUIRED_ATOM_COLUMNS](usm/core/model.py:53).
  - Dtype enforcement and schema normalization happen in [USM.__post_init__](usm/core/model.py:97).

- Bonds table (optional)
  - Columns (name: dtype)
    - bid: Int32; dense 0..M-1
    - a1: Int32; atom endpoint (aid)
    - a2: Int32; atom endpoint (aid) with invariant a1 < a2
    - order: float32 (e.g., 1.0, 1.5)
    - type: string (optional future use)
    - source: string (e.g., "mdf.connections")
    - order_raw: string (original token if present, e.g. "1.5")
    - mol_index: Int32 (scoping/debug)
    - notes: string
  - Dtype and normalization in [USM.__post_init__](usm/core/model.py:107).

- Molecules table (optional v0.1)
  - Columns (name: dtype)
    - mid: Int32; dense 0..K-1
    - mol_label: string
    - mol_index: Int32
    - mol_block_name: string
    - provenance: string

- Cell & PBC metadata
  - Stored in USM.cell (dict)
    - pbc: bool
    - a, b, c: float64 (Å)
    - alpha, beta, gamma: float64 (deg)
    - spacegroup: string
  - Access example: [USM.cell](usm/core/model.py:93)

- Provenance & preserved text
  - Stored in USM.provenance (dict) and USM.preserved_text (dict)
  - Round-trip fidelity relies on preserved_text:
    - car_header_lines, car_footer_lines for CAR
    - mdf_header_lines, mdf_footer_lines, mdf_molecule_order for MDF
  - Set by importers: [load_car()](usm/io/car.py:128), [load_mdf()](usm/io/mdf.py:220)

ID and ordering semantics
- Atom ids (aid): dense Int32 row index; recomputed on any operation that changes rows or order. Enforced by [USM.__post_init__](usm/core/model.py:101).
- Bond ids (bid): dense Int32 row index; endpoints normalized such that a1 < a2; recomputed when bonds are remapped. Enforced by [USM.__post_init__](usm/core/model.py:111).
- Molecule ids (mid): optional; can be derived deterministically by first-appearance of (mol_label, mol_index, mol_block_name) via [renumber_molecules()](usm/ops/renumber.py:33).

Units and numeric precision
- Coordinates: Å (float64)
- Angles: deg (float64)
- Charges: e (float32)
- Bundle manifest records units and tolerances in [save_bundle()](usm/bundle/io.py:42).

Format normalization
- CAR import preserves header/footer; reads atom lines of form:
  name x y z mol_label mol_index atom_type element charge
  via [ATOM_LINE_RE](usm/io/car.py:17) and [load_car()](usm/io/car.py:128).
- MDF import preserves @columns, !Date, @molecule order, and connections (raw tokens). Topology columns parsed via [MDF_LINE_RE](usm/io/mdf.py:9) and [load_mdf()](usm/io/mdf.py:220). Bonds normalized from connections in [_build_bonds_from_connections()](usm/io/mdf.py:148).

Determinism and validation
- All ops return deterministic ordering and id remaps. See selectors and transforms in:
  - [select_by_element()](usm/ops/select.py:55), [select_box()](usm/ops/select.py:80), [select_within_radius()](usm/ops/select.py:94)
  - [translate()](usm/ops/transform.py:16), [rotation_matrix_from_axis_angle()](usm/ops/transform.py:32), [rotate()](usm/ops/transform.py:56), [scale()](usm/ops/transform.py:71), [wrap_to_cell()](usm/ops/transform.py:86)
  - [replicate_supercell()](usm/ops/replicate.py:17)
  - [merge_structures()](usm/ops/merge.py:25), [compose_on_keys()](usm/ops/compose.py:12)
  - [renumber_atoms()](usm/ops/renumber.py:17), [renumber_molecules()](usm/ops/renumber.py:33)
- Schema and numeric checks in [USM.validate_basic()](usm/core/model.py:134).

Serialization (USM Bundle)
- Folder of tables + manifest.json; Parquet preferred, CSV fallback when no Parquet engine is available:
  - Save: [save_bundle()](usm/bundle/io.py:42)
  - Load: [load_bundle()](usm/bundle/io.py:101)
- Manifest records per-table dtypes, cell, provenance, preserved_text, and tolerances.

Known interoperability behaviors
- connections_raw (MDF) preserved verbatim for lossless round-trip; normalized bonds represent best-effort inference.
- CAR/MDF exporters default to preserved header/footer when available:
  - CAR writer: [save_car()](usm/io/car.py:188)
  - MDF writer: [save_mdf()](usm/io/mdf.py:332)