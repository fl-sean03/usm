Unified Structure Model (USM) v0.1 — Design
Overview
USM provides a consistent, high-fidelity in-memory model for Materials Studio CAR/MDF structures, with direct import/export and deterministic structural operations. Initial focus targets WAT and DOP examples using Pandas + PyArrow + NumPy on CPython 3.10+.
Repository context
This repo includes example inputs in [msiExamples](msiExamples) and wrapper plugins that currently shell out to external scripts (see [parsers/car/plugin.py](parsers/car/plugin.py), [parsers/mdf/plugin.py](parsers/mdf/plugin.py), [writers/car/plugin.py](writers/car/plugin.py), [writers/mdf/plugin.py](writers/mdf/plugin.py), [writers/pdb/plugin.py](writers/pdb/plugin.py)). USM will implement direct in-repo importers/exporters and provide a self-contained library.
Goals
- Ingest MDF/CAR into a canonical schema capturing atoms, optional bonds, group labels, cell/PBC, and provenance.
- Provide deterministic operations (selection, transforms, replication, merge, renumber).
- Round-trip to CAR/MDF while preserving critical formatting (headers/footers and MDF @columns/!Date).
- Serialize to a “USM Bundle” (Parquet tables + JSON manifest) with stable schema/versioning.
Non-goals (v0.1)
- Advanced chemical perception, force-field assignment, crystallography pipelines, GUIs.
- Multi-symmetry or advanced spacegroup handling beyond passthrough preservation.
Core schema (atoms, bonds, groups, metadata)
Atoms table (required)
- aid int32: dense row id (equals row order)
- name string: atom name (e.g., H1, C7)
- element string
- atom_type string
- charge float32
- x, y, z float64 (Angstrom)
- mol_label string (e.g., MOL2, XXXX)
- mol_index int32 (e.g., 1 from MOL2_1 or XXXX 1)
- mol_block_name string (from MDF @molecule; empty for CAR)
- MDF-compatible preservation fields (nullable): isotope, formal_charge, switching_atom, oop_flag, chirality_flag, occupancy, xray_temp_factor, charge_group, connections_raw
Bonds table (optional)
- bid int32; a1, a2 int32 (a1 < a2 invariant); order float32; type string; source string; order_raw string; mol_index int32; notes string
Molecules table (optional v0.1)
- mid int32; mol_label string; mol_index int32; mol_block_name string; provenance string
Cell and PBC metadata
- pbc bool; a, b, c float64; alpha, beta, gamma float64; spacegroup string
Provenance and preserved text
- source_format, source_path, date_line, parse_notes
- preserved_text: car_header_lines, car_footer_lines, mdf_header_lines, mdf_molecule_order
ID and ordering policy
- Atoms: aid equals row order; reallocated deterministically after any change.
- Bonds: bid equals row order; endpoints map to current aids; maintain a1 < a2.
- Molecules: mid assigned by first appearance of composite (mol_label, mol_index, mol_block_name).
Table engine and units
- Pandas DataFrames with NumPy-backed columns; PyArrow for Parquet IO.
- Units: Angstrom for coordinates, degrees for angles, e for charges. Internal xyz and cell as float64; charge as float32.
USM Bundle format
Layout (directory)
- atoms.parquet, bonds.parquet (optional), molecules.parquet (optional), residues.parquet (optional)
- manifest.json
Manifest keys
- version: "0.1"
- units and numeric_tolerances (coord_abs=1e-5, cell_abs=1e-8)
- required_columns and dtypes per table
- cell metadata
- provenance and preserved_text blocks
- extras reserved for forward compatibility
Importers (direct, no subprocess)
CAR importer [usm/io/car.py](usm/io/car.py)
- Parse header lines (including PBC=ON/OFF, !DATE, optional PBC a b c α β γ line), preserve them verbatim.
- Parse atom lines with regex tolerant to spacing: name, x, y, z, mol_label, mol_index, atom_type, element, charge.
- Stop at first end line; preserve trailing footer lines verbatim.
MDF importer [usm/io/mdf.py](usm/io/mdf.py)
- Preserve @column definitions, !Date, @molecule order.
- Parse name prefix "MOL2_1:C1" into mol_label, mol_index, name; map columns 1–12; store connections_raw exactly.
- Normalize bonds from connections when unambiguous (token "ATOM[/order]"); otherwise keep in raw form and note ambiguity.
Exporters
CAR exporter [usm/io/car.py](usm/io/car.py)
- If preserved header/footer exist, write byte-for-byte; else synthesize canonical header/footer.
- Render atom rows in stable order with consistent numeric formatting; xyz 9 decimals, charge 3 decimals.
MDF exporter [usm/io/mdf.py](usm/io/mdf.py)
- If preserved header exists, write byte-for-byte; else synthesize with canonical @column block and !Date line.
- Emit @molecule blocks; for connections, prefer lossless using connections_raw; normalized mode available via flag.
Operations (v0.1)
Selection [usm/ops/select.py](usm/ops/select.py)
- By element, name, mol_label/mol_index, aid ranges, spatial box or radius.
Transforms [usm/ops/transform.py](usm/ops/transform.py)
- translate, rotate (origin/centroid), scale, wrap_to_cell (when pbc)
Supercell [usm/ops/replicate.py](usm/ops/replicate.py)
- Tile by integers (na, nb, nc); duplicate bonds; deterministic reindex.
Merge/append and renumber [usm/ops/merge.py](usm/ops/merge.py), [usm/ops/renumber.py](usm/ops/renumber.py)
- Concatenate with metadata reconciliation; stable renumbering policies.
Validation [usm/core/validate.py](usm/core/validate.py)
- Schema/dtype checks; numeric finiteness; PBC consistency.
Determinism and numeric tolerances
- Re-running identical ops yields identical Parquet tables and manifest JSON (ignoring timestamps).
- Expected coordinate tolerance on round-trip ≤ 1e-5 Å.
WAT and DOP specifics
WAT CAR [msiExamples/WAT.car](msiExamples/WAT.car)
- PBC=OFF; 3 atoms; two trailing end lines; store header/footer verbatim; no bonds.
WAT MDF [msiExamples/WAT.mdf](msiExamples/WAT.mdf)
- One molecule block; connections show O1 connected to H1/H2; parse as bonds but preserve connections_raw.
DOP CAR [msiExamples/DOP.car](msiExamples/DOP.car)
- PBC=OFF; 23 atoms; aromatic ring present via types; no bonds in CAR.
DOP MDF [msiExamples/DOP.mdf](msiExamples/DOP.mdf)
- connections include fractional order "1.5" for aromatics; parse to order=1.5; retain order_raw.
API surface (initial)
- load_car(path) and save_car(usm, path, preserve_headers=True) in [usm/io/car.py](usm/io/car.py)
- load_mdf(path) and save_mdf(usm, path, preserve_headers=True, write_normalized_connections=False) in [usm/io/mdf.py](usm/io/mdf.py)
- save_bundle(usm, folder) and load_bundle(folder) in [usm/bundle/io.py](usm/bundle/io.py)
- select_atoms, translate, rotate, scale, replicate_supercell, merge, renumber in [usm/ops/*](usm/ops)
Minimal implementation order
1) Core model and validation in [usm/core/model.py](usm/core/model.py)
2) CAR importer/exporter for WAT (PBC OFF), plus tests
3) MDF importer for WAT/DOP with connections parsing; preserve connections_raw
4) MDF exporter; tests
5) USM Bundle save/load (Parquet+JSON)
6) Operations (selection, transforms, replication) with tests
7) Round-trip tests and examples
Acceptance criteria (v0.1)
Import
- Atom counts match; required columns/dtypes present; PBC/cell captured when present.
Round-trip
- CAR: header/footer preserved exactly; xyz diffs ≤ 1e-5 Å; atom count identical.
- MDF: !Date and @column preserved; molecule blocks preserved; connections preserved (raw tokens).
Operations
- Deterministic outputs; replication multiplies counts by na*nb*nc; transforms match analytic expectations.
Performance
- Linear memory; WAT/DOP import and simple ops complete in sub-second wall time on commodity hardware.
Risks and mitigations
- External script path coupling in existing wrappers: mitigated by direct importers/exporters.
- Ambiguous MDF connections: preserve raw tokens and annotate parse notes; do not force lossy inference.
- Formatting drift: header/footer preservation; consistent number formatting on writers.
Files to add (initial)
- [usm/core/model.py](usm/core/model.py), [usm/io/car.py](usm/io/car.py), [usm/io/mdf.py](usm/io/mdf.py), [usm/bundle/io.py](usm/bundle/io.py), [tests/](tests)
Examples
- Load WAT CAR, translate by [1,0,0], save CAR and MDF. Load DOP MDF, select aromatic ring, rotate 90°, save bundle.
Documentation plan
- This DESIGN file plus [DATA_MODEL.md](docs/DATA_MODEL.md), [API.md](docs/API.md), [EXAMPLES.md](docs/EXAMPLES.md), [PERFORMANCE.md](docs/PERFORMANCE.md), [LIMITS.md](docs/LIMITS.md) to be added as implementation lands.