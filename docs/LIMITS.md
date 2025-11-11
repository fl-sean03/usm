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
  - PDB exporter is minimal (ATOM/TER/END; optional CRYST1)
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
- When using CSV fallback, loading infers dtypes via USM schema coercion; string columns and Int32 columns remain compatible

Determinism and IDs
- All operations (selection, transforms, merge, replicate, renumber) produce deterministic outputs and contiguous IDs
- compose_on_keys depends on the uniqueness of key tuples (mol_label, mol_index, name); duplicates are resolved by last-wins semantics when joining

Performance and scale
- Designed to scale linearly to ~1e5 atoms with typical operations
- String-heavy columns (e.g., many unique names or types) can increase memory; consider reducing cardinality or using categorical encodings in future versions

Error handling and fallbacks
- Plugin wrappers (parse/write) prefer direct in-repo readers/writers; fallback to external scripts only if required
- When Parquet engines are missing, bundle save/load uses CSV fallback automatically

Future roadmap (shortlist)
- Full triclinic lattice support (fractional coordinate transforms) for wrap/replicate
- Additional operations: symmetry expansion, constrained selections, map/unmap fractional <-> Cartesian
- Richer PDB exporter (MODEL/ENDMDL; altLoc, occupancy/tempFactor formatting; CONECT from bonds)
- Optional polars/pyarrow interchange for large-scale performance
- Additional text format importers (e.g., XYZ, GRO) and writers as needed
- Improved MDF exporter with configurable formatting profiles to better match downstream tool expectations