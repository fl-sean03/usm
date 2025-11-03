USM v0.1 — Performance and Scale Probe

Goals
- Scale comfortably to mid-sized systems (~1e5 atoms) with predictable, linear memory usage
- Deterministic, vectorized operations for typical workflows (select, transform, compose/merge, replicate)
- Portable I/O with Parquet when available; automatic CSV fallback when not

Table engine
- Pandas + NumPy for in-memory tables and vectorized math
- Parquet via PyArrow or fastparquet; if not installed, USM falls back to CSV automatically in the bundle

Complexity summary (typical operations)
- Import (CAR, MDF): O(N) over atoms; MDF bonds inference is O(E) in number of connection tokens
- Export (CAR, MDF): O(N) with string formatting
- Selection (element, name, box, radius): O(N) filter; box/radius uses vectorized comparisons and L2 distances via NumPy
- Transforms (translate, rotate, scale): O(N) matrix ops; wrap_to_cell O(N)
- Merge (k structures): O(sum N_i) for concatenation; bonds mapped once per input; stable remap O(E)
- Compose on keys: O(N log N) for table joins (Pandas merges on string+int keys)
- Replicate: O(N * product(na, nb, nc)) to materialize tiled images
- Renumber: O(N log N) if sorting; otherwise O(N)

Memory model
- Atoms table: roughly 8 bytes per float64 column, 4 bytes per float32, and dictionary-encoded strings in Pandas varying by content
- Practical rule of thumb: ~120–180 bytes per atom for typical columns (x,y,z, element, type, name, mol labels, charge)
- Bonds table: ~32–48 bytes per bond depending on dtype mix
- Example: 1e5 atoms with atoms-only structure ≈ 12–18 MB plus Python overhead; double if many string columns are unique

Parquet vs CSV
- Parquet is recommended for speed and compactness
- If pyarrow/fastparquet are unavailable, save_bundle/load_bundle transparently use CSV (see [save_bundle()](usm/bundle/io.py:42))

Scale probe (script)
- A reference script is provided at [examples/wat_dop_pipeline.py](examples/wat_dop_pipeline.py) for end-to-end demo, and [scripts/scale_probe.py](scripts/scale_probe.py) for scale measurements on synthetic tilings
- The probe:
  - Loads WAT (3 atoms) from CAR
  - Assigns an orthorhombic cell
  - Replicates to ~1e5 atoms (e.g., 33x32x32 × 3 ≈ 101,376 atoms)
  - Runs selection, translate, rotate
  - Optionally writes a bundle (Parquet if available; CSV otherwise)
  - Prints wall-clock timings

Expected timings (guidance; depends on CPU, Python build, and BLAS)
- Import CAR (3 atoms; trivial): < 5 ms
- Replicate to ~1e5 atoms: ~0.1–0.5 s
- Selection (element, box, radius): ~5–50 ms
- Translate/rotate/scale: ~5–50 ms
- save_bundle/load_bundle:
  - Parquet (pyarrow): ~50–400 ms for 1e5 atoms
  - CSV fallback: ~2–3x slower and larger on disk

Recommendations
- Install pyarrow for faster bundle I/O: pip install pyarrow
- Avoid excessive, repeated DataFrame copies; use in_place=True for transforms only when it is safe
- For large merges, pre-filter and minimize string cardinality where possible (e.g., categorical columns)
- Replicate last; apply transforms before replication to reduce compute

Known performance trade-offs
- Current wrap_to_cell assumes orthorhombic cells; non-orthorhombic wrap would require fractional conversion (future work)
- Bonds inference uses simple parsing; extremely large molecules with dense connections will scale with number of tokens (O(E))

Repro steps
- Run the scale probe:
  python scripts/scale_probe.py --na 33 --nb 32 --nc 32 --bundle out/scale_bundle
- Example output includes:
  - replicate time, selection time, transform time
  - save_bundle time and format (parquet or csv)