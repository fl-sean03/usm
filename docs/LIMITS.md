USM v0.1 — Known Limits and Future Work

Scope and non-goals (v0.1)
- Chemical perception not included
  - No aromaticity detection beyond MDF connections parsing (order=1.5 where provided)
  - No valence checks, bond typing heuristics, or force-field parameterization
- Crystallography
  - General triclinic lattice support for wrap_to_cell and replicate_supercell via fractional coordinates
  - Orthorhombic behaviors preserved; float64 math; frac↔xyz round-trip property tests ≤ 1e-12
- Formats
  - Primary focus on Materials Studio CAR/MDF
  - CIF support is intentionally minimal:
    - Supports cell parameters + a single `atom_site` loop with fractional coords (see [`load_cif()`](src/usm/io/cif.py:222))
    - CIF export writes cell + `atom_site` fractional coords (see [`save_cif()`](src/usm/io/cif.py:365))
    - Symmetry expansion is not implemented (`expand_symmetry=False` only); disorder models and rich crystallographic metadata are out of scope for v0.1
  - PDB exporter: ATOM/TER/END with optional CRYST1; opt-in CONECT (from bonds) and single-model MODEL/ENDMDL framing are available. Defaults preserve prior behavior (no CONECT, no MODEL).
- GUI or interactive visualization not part of scope

I/O and round-trip fidelity
- CAR
  - Byte-accurate preservation of header/footer achieved; formatted numbers may differ in whitespace but retain analytical values
- MDF
  - @column and !Date lines preserved when present (lossless mode)
  - connections_raw tokens preserved verbatim; normalized export available by flag
  - formal_charge tokens preserved without truncation; writer emits tokens verbatim (e.g., 1/2+) using single-space field delimiters, and right-aligns standard-width values to width 3 for backward compatibility
  - Coordinates not stored in MDF; USM retains NaN for xyz fields from MDF-only imports

Bundles and portability
- Bundles prefer Parquet; CSV fallback used automatically if no Parquet engine is available
- CSV fallback: writer emits numeric columns with high precision (float_format="%.17g") for deterministic numeric round-trips; loader coerces dtypes via USM schema so string and nullable Int32 columns remain compatible
- Load precedence when both formats exist: prefer Parquet if readable; otherwise fall back to CSV

Determinism and IDs
- All operations (selection, transforms, merge, replicate, renumber) produce deterministic outputs and contiguous IDs
- compose_on_keys depends on the uniqueness of key tuples (mol_label, mol_index, name); duplicates are resolved by last-wins semantics when joining
- Composition diagnostics (v0.1+):
  - Coverage metrics over unique key tuples: matched_count, left_only_count, right_only_count, primary_total, secondary_total, coverage_ratio
  - Policy handling: policy="silent" (default) attaches metrics under provenance["compose_coverage"]; policy="warn" also appends a deterministic message to provenance.parse_notes when coverage is below coverage_threshold; policy="error_below_coverage" raises ValueError below threshold
  - Default coverage_threshold=0.95; configurable by callers (e.g., workspace runner)
  - Metrics and messages are deterministic and stable across runs for identical inputs
Performance and scale
- Designed to scale linearly to ~1e5 atoms with typical operations
- String-heavy columns (e.g., many unique names or types) can increase memory; consider reducing cardinality or using categorical encodings in future versions

Error handling and fallbacks
- Plugin wrappers (parse/write) prefer direct in-repo readers/writers; fallback to external scripts only if required
- When Parquet engines are missing, bundle save/load uses CSV fallback automatically

Future roadmap (shortlist)
- Full triclinic lattice support (fractional coordinate transforms) for wrap/replicate
- Additional operations: symmetry expansion, constrained selections, map/unmap fractional <-> Cartesian
- PDB exporter roadmap: multi-model trajectories; additional record types (e.g., ANISOU, SSBOND, HELIX/SHEET); serial overflow handling (> 99999); refined atom-name justification per element.
- Optional polars/pyarrow interchange for large-scale performance
- Additional text format importers (e.g., XYZ, GRO) and writers as needed
- Improved MDF exporter with configurable formatting profiles to better match downstream tool expectations