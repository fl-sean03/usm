USM v0.1 API

Overview
The Unified Structure Model (USM) normalizes Materials Studio CAR and MDF into a single, in-memory representation (tables + metadata), supports deterministic operations, and round-trips back to CAR/MDF (and optional PDB). The table engine is Pandas + NumPy; Parquet is preferred for the bundle format with automatic CSV fallback.

Core modules
- I/O
  - [usm/io/car.py](usm/io/car.py) — CAR import/export
  - [usm/io/mdf.py](usm/io/mdf.py) — MDF import/export
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

- PDB
  - save_pdb(usm: USM, path: str) -> str
    - Minimal PDB writer (ATOM, TER, END). Writes CRYST1 when pbc is True and all parameters are finite (orthorhombic assumed).
    - Derives residue fields if absent (mol_block_name -> resName, mol_index -> resSeq, default chain A).

- Bundle I/O
  - save_bundle(usm: USM, folder: str) -> str
    - Save USM to a folder with atoms/bonds/molecules and a manifest.json. Tries Parquet; falls back to CSV automatically if no Parquet engine is available.
  - load_bundle(folder: str) -> USM
    - Load a bundle written by save_bundle. Supports both Parquet and CSV.

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
- Bundle portability: Parquet recommended (pyarrow/fastparquet), with automatic CSV fallback.

Examples
See [docs/EXAMPLES.md](docs/EXAMPLES.md) for WAT and DOP end-to-end workflows, and [examples/wat_dop_pipeline.py](examples/wat_dop_pipeline.py) for a runnable script using the above APIs.

Tests as reference
- CAR round-trip: [tests/test_car_roundtrip.py](tests/test_car_roundtrip.py)
- MDF import/round-trip: [tests/test_mdf_import.py](tests/test_mdf_import.py), [tests/test_mdf_roundtrip.py](tests/test_mdf_roundtrip.py)
- Operations/transform/replicate/compose/merge/bundle: [tests/test_ops_transforms.py](tests/test_ops_transforms.py), [tests/test_ops_wrap_and_renumber.py](tests/test_ops_wrap_and_renumber.py), [tests/test_ops_suite.py](tests/test_ops_suite.py)