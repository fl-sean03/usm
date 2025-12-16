USM v0.1 API

Overview
The Unified Structure Model (USM) normalizes Materials Studio CAR and MDF into a single, in-memory representation (tables + metadata), supports deterministic operations, and round-trips back to CAR/MDF (and optional PDB). The table engine is Pandas + NumPy; Parquet is preferred for the bundle format with automatic CSV fallback.

Core modules
- I/O
  - [usm/io/__init__.py](usm/io/__init__.py) — Optional convenience aggregator (centralized imports)
  - [usm/io/car.py](usm/io/car.py) — CAR import/export
  - [usm/io/mdf.py](usm/io/mdf.py) — MDF import/export
  - [usm/io/cif.py](usm/io/cif.py) — CIF import/export (minimal, dependency-free)
  - [usm/io/pdb.py](usm/io/pdb.py) — PDB export (best-effort)
  - [usm/bundle/io.py](usm/bundle/io.py) — USM bundle save/load (Parquet preferred, CSV fallback)
- Core model/validation
  - [usm/core/model.py](usm/core/model.py)
- Operations
  - [usm/ops/select.py](usm/ops/select.py)
  - [usm/ops/transform.py](usm/ops/transform.py)
  - [usm/ops/replicate.py](usm/ops/replicate.py)
  - [usm/ops/merge.py](usm/ops/merge.py)
  - [usm/ops/compose.py](usm/ops/compose.py)
  - [usm/ops/renumber.py](usm/ops/renumber.py)
  - [usm/ops/requirements.py](usm/ops/requirements.py) — Deterministically derive v0.1 `requirements.json` (atom/bond/angle types) from a USM structure.

Data model summary
See [docs/DATA_MODEL.md](docs/DATA_MODEL.md) for the full schema (columns/dtypes), cell/PBC metadata, preservation strategy, and ID policies.

API reference (selected)
- CAR
  - load_car(path: str) -> USM
    - Parse CAR file, preserving header/footer lines and PBC cell line if present.
    - Returns a USM with atoms populated; bonds empty.
  - save_car(usm: USM, path: str, preserve_headers: bool = True) -> str
    - Write a CAR file from USM. Uses preserved header/footer if available; otherwise synthesizes a canonical header/footer.

- MDF
  - load_mdf(path: str) -> USM
    - Parse MDF topology rows and @molecule sections.
    - Preserve @column block and !Date in preserved_text; capture connections_raw and normalize to a bonds table.
    - Atoms have NaN coordinates by design (MDF does not carry coordinates).
  - save_mdf(usm: USM, path: str, preserve_headers: bool = True, write_normalized_connections: bool = False) -> str
    - Write MDF with lossless connections (connections_raw) when available; normalized tokens when write_normalized_connections=True.
    - Preserved header/footer written byte-for-byte when present.

- CIF (minimal)
  - load_cif(path: str, *, mol_label: str="XXXX", mol_index: int=1, mol_block_name: Optional[str]=None, expand_symmetry: bool=False) -> USM
    - Parse lattice parameters and an atom_site loop containing fractional coordinates.
    - Converts fractional -> Cartesian via USM lattice helpers.
    - Symmetry expansion is not implemented (must remain False).
  - save_cif(usm: USM, path: str, *, data_block_name: Optional[str]=None, spacegroup: Optional[str]=None, wrap_frac: bool=True) -> str
    - Write a minimal CIF with cell parameters + atom_site fractional coordinates computed from USM xyz.
    - Does not write symmetry operations; writes P 1 when no spacegroup is available.

- PDB
  - save_pdb(usm: USM, path: str, include_conect: bool = False, include_model: bool = False, model_index: int = 1, conect_policy: str = "dedup") -> str
    - Minimal PDB writer (ATOM, TER, END). Writes CRYST1 when pbc is True and all lattice parameters are finite. Optional CONECT from bonds and single-model MODEL/ENDMDL framing.
    - Flags:
      - include_conect: write CONECT records derived from usm.bonds (a1,a2 mapped to serial=aid+1); neighbors sorted; max 4 per line; chunk as needed.
      - conect_policy: "dedup" (default; emit bond once on lower-serial) or "full" (both directions).
      - include_model: wrap records in MODEL {model_index} ... ENDMDL; CRYST1 appears after MODEL when present.
    - Deterministic output ordering and formatting; ATOM element/name justification unchanged.

- Bundle I/O
  - save_bundle(usm: USM, folder: str) -> str
    - Save USM to a folder with atoms/bonds/molecules and a manifest.json. Tries Parquet; falls back to CSV automatically if no Parquet engine is available.
    - CSV fallback writes numeric columns with high precision (float_format="%.17g") for deterministic numeric round-trips.
  - load_bundle(folder: str) -> USM
    - Load a bundle written by save_bundle. Supports both Parquet and CSV.
    - Precedence: prefers Parquet when present/readable; falls back to CSV. If both exist, Parquet wins by default.
    - Strict manifest handling with clear ValueError messages for missing/unknown version, missing files, or row-count mismatches.
    - Deterministic: preserves stored row order; [USM.__post_init__](src/usm/core/model.py:98) enforces schema/dtypes and contiguous IDs; [USM.validate_basic()](src/usm/core/model.py:135) sanity-checks required fields.

- Selection (return new USM with deterministically reindexed aids)
  - select_by_element(usm, elements: Iterable[str]) -> USM
  - select_by_name(usm, names: Iterable[str]) -> USM
  - select_by_molecule_index(usm, mol_index: int) -> USM
  - select_box(usm, xmin, xmax, ymin, ymax, zmin, zmax) -> USM
  - select_within_radius(usm, center: Iterable[float], radius: float) -> USM

- Transforms (in-place optional via in_place=True)
  - translate(usm, delta: Iterable[float], in_place: bool=False) -> USM
  - rotation_matrix_from_axis_angle(axis: Iterable[float], angle_deg: float) -> np.ndarray
  - rotate(usm, R: np.ndarray, origin=(0,0,0), in_place: bool=False) -> USM
  - scale(usm, factors: Union[float, Iterable[float]], origin=(0,0,0), in_place: bool=False) -> USM
  - wrap_to_cell(usm, in_place: bool=False) -> USM
    - Orthorhombic wrapping only (angles ~ 90 deg). No-op when PBC False or cell params missing.

- Replication / Merge / Composition / Renumber
  - replicate_supercell(usm, na: int, nb: int, nc: int, add_image_indices: bool=True) -> USM
    - Requires orthorhombic cell with finite parameters.
  - merge_structures(usms: List[USM], cell_policy: str="strict") -> USM
    - Concatenate multiple USMs; remap bonds; reconcile cell via policy "strict"|"first"|"error".
  - compose_on_keys(primary: USM, secondary: USM, keys: List[str]=["mol_label","mol_index","name"]) -> USM
    - Join atoms by keys, filling missing columns in primary from secondary; bonds prefer secondary when present.
  - renumber_atoms(usm, order_by: Optional[List[str]]=None, in_place: bool=False) -> USM
    - Deterministically reassign aids via stable sort. Defaults to ["mol_index","name"] when present.
  - renumber_molecules(usm, in_place: bool=False) -> USM
    - Derive atom-level "mid" (first appearance order by (mol_label, mol_index, mol_block_name)).

Usage notes
- Determinism: All operations are stable and deterministic for identical inputs.
- Dtype enforcement: Constructing/returning USM normalizes schema with fixed dtypes and contiguous aids/bids.
- Coordinates: Always float64 in Å internally; writer format may use fixed-width decimals.
- Bonds from MDF: Normalized undirected bonds are constructed; original tokens are preserved in connections_raw.
- CIF: Intended for minimal import/export of cell + atom_site positions. Advanced crystallography (symmetry expansion, disorder models) is out of scope for v0.1.
- Bundle portability: Parquet recommended (pyarrow/fastparquet), with automatic CSV fallback.

Examples
See [docs/EXAMPLES.md](docs/EXAMPLES.md) for WAT and DOP end-to-end workflows, and [examples/wat_dop_pipeline.py](examples/wat_dop_pipeline.py) for a runnable script using the above APIs.

Tests as reference
- CAR round-trip: [tests/test_car_roundtrip.py](tests/test_car_roundtrip.py)
- MDF import/round-trip: [tests/test_mdf_import.py](tests/test_mdf_import.py), [tests/test_mdf_roundtrip.py](tests/test_mdf_roundtrip.py)
- Operations/transform/replicate/compose/merge/bundle: [tests/test_ops_transforms.py](tests/test_ops_transforms.py), [tests/test_ops_wrap_and_renumber.py](tests/test_ops_wrap_and_renumber.py), [tests/test_ops_suite.py](tests/test_ops_suite.py)