# MDF Parser Audit — PBC image-offset preservation and beyond

**Date**: 2026-05-31
**Scope**: `usm.io._mdf_parser`, `usm.io.mdf`, `usm.ops.mdf_conn_preserve`, `usm.core.model.USM` (bonds schema only)
**Trigger**: A downstream consumer (`iff-registry`) discovered that USM was silently dropping the Materials Studio per-token periodic-image suffix (`%abc#i`) on every MDF connection, causing all PBC bonds to be returned with shift `(0, 0, 0)`. A renderer downstream then drew straight cylinders spanning whole unit cells. iff-registry PR #55 worked around the bug by re-parsing `connections_raw` outside USM; this audit closes the loop by fixing the parser and propagating the recovered offsets through the bond model.

## 1. Background — the wire format

A Materials Studio MDF connection token (one of the space-separated entries on the tail of an atom line) has the shape:

```
TARGET[%abc#i][/order]
```

Examples drawn from Sean's real MDF corpus (`/home/sfhs1/Backups/Dropbox/MANUSCRIPT_Al2O3/.../Gibbsite_g_Al(OH)3_unit_cell_Saalfeld_1974.mdf`):

```
XXXX_1:Al1  ...  O1 O4 O5 O6%110#3 O4%010#3 O2%110#3
XXXX_1:Al2  ...  O1 O3 O5%0-10#1 O2%110#3 O3%100#3 O6%100#3
XXXX_1:O2   ...  H2 Al1%110#3 Al2%110#3
```

- `TARGET` is either a bare atom name within the source atom's `(mol_label, mol_index)` scope (e.g. `O1`) or fully qualified `LABEL_INDEX:NAME` (e.g. `XXXX_1:Al1`).
- `%abc#i` is the periodic-image qualifier:
  - `a`, `b`, `c` are each a **single signed digit** representing the shift along the corresponding cell axis (a, b, c).
  - `i` is the image index (Materials Studio's internal bookkeeping; positive integer).
- `/order` is the bond order; absent ⇒ order = 1.0.

A complete corpus scan of every `.mdf` under `~/Backups/Dropbox` found only six distinct suffix patterns in the wild:

```
%0-10#1   →  (0, -1, 0)
%010#1    →  (0,  1, 0)
%010#3    →  (0,  1, 0)
%100#3    →  (1,  0, 0)
%110#3    →  (1,  1, 0)
%-1-10#5  →  (-1, -1, 0)
```

All match the regex `%(-?\d)(-?\d)(-?\d)#\d+`. The image index `#i` carries no geometric information for our purposes; it is internal Materials Studio bookkeeping. We **preserve `i` only on the raw text** (already done via `connections_raw`); the parsed shift drops it.

## 2. Current state — file:line tour

### 2.1 `usm.io._mdf_parser.build_bonds_from_connections` — the source of the bug

```python
# usm/io/_mdf_parser.py:188-190 (BEFORE FIX)
# Strip any Materials Studio constraint suffix after '%' on the base token
if "%" in base:
    base = base.split("%", 1)[0]
```

The docstring at line 162 explicitly documents the behavior as intentional:

```
- Optional Materials Studio suffixes after '%' are ignored
```

The result: every parsed `Bond` has `ix = iy = iz = 0`, regardless of what was on the wire. The Bond schema (`BONDS_DTYPES` in `core/model.py:38`) already has `ix`, `iy`, `iz` columns with `Int32` dtype — the infrastructure to carry the shift exists, but the producer never populates it.

### 2.2 `usm.core.model.USM.__post_init__` — already ready

The Bond model already understands image flags. It auto-defaults missing `ix/iy/iz` to 0 (`model.py:118-122`) and even negates them on `a1 ↔ a2` swap (`model.py:126-131`) and renormalizes self-bonds (`model.py:133-145`). The downstream `perceive_periodic_bonds` and renumbering ops also flow `ix, iy, iz` end-to-end. **The model is ready; the MDF reader is the only blocker.**

### 2.3 `usm.ops.mdf_conn_preserve` — name-only preservation

`parse_base_token` (line 59) returns only the base `LABEL_INDEX:NAME` and discards the `/order` and `%...` suffixes by design — it's a name-stripping helper for `cleanse_connections_raw`, which removes references to deleted atoms. **This usage is correct** (we only need the partner identity to decide whether to drop the token); no change required.

### 2.4 `usm.io.mdf.save_mdf` and `_compose_connections_for_atom` — the writer

The writer has two modes:
- **Preserve mode (`write_normalized_connections=False`, default)**: writes `connections_raw` verbatim, so a load → save round-trip is byte-identical for the connections field. PBC suffixes are preserved through this path *without* the parser fix because the raw string is the source of truth.
- **Normalized mode**: rebuilds connections from `(adj_list, aid_to_info, /order)`. Does **not** currently emit `%abc#i` suffixes. After the parser fix this means: **load(PBC MDF) → save(normalized=True) loses the PBC shifts unless the writer is taught to emit suffixes from `(ix, iy, iz)`.** This is a real (though scoped) regression risk and is fixed in this PR.

## 3. Findings — every information-loss site

| # | Site | Severity | Disposition |
|---|------|----------|-------------|
| F1 | `build_bonds_from_connections` strips `%abc#i` and discards the shift | **HIGH** (the trigger) | **Fix in this PR.** Parse the suffix into `(ix, iy, iz)` and emit on the Bond row. |
| F2 | Normalized-mode writer drops `(ix, iy, iz)` | **MEDIUM** (only affects `write_normalized_connections=True`; default is preserve mode) | **Fix in this PR.** Emit `%abc#i` (using `i=1` as a default index) when any of `ix/iy/iz` is non-zero. |
| F3 | `parse_atom_line` ignores the image index `i` from `%abc#i` | LOW (no geometric info; only Materials Studio bookkeeping) | **Defer.** Document explicitly. The raw string is preserved on `connections_raw` so any tool that needs `#i` can recover it. |
| F4 | Self-bonds (`tgt_aid == src_aid` with non-zero shift) are silently skipped | **MEDIUM** | **Fix in this PR.** A self-image bond (`A — A%100#1`) is a legitimate PBC bond and must be emitted. The Bond model already self-normalizes self-bonds (`model.py:133-145`). |
| F5 | Tokens that don't resolve (unknown partner) are silently skipped | LOW | **Defer with note.** Existing behavior; could log a warning but changing it is out of scope. Documented in the docstring. |
| F6 | `tgt_prefix_re` accepts only `[A-Za-z0-9]+` for label (no underscore inside label) | LOW | **Defer.** No real-world MDFs exhibit underscored labels and changing it could mask real parse errors. Filed as future work. |
| F7 | `MDF_LINE_RE.match` uses `match` (anchored at start) and the parse_atom_line raises on a malformed line, but `load_mdf` *also* pre-screens with `MDF_LINE_RE.match` and silently skips non-matches | LOW | **Defer.** Existing tolerant behavior; switching to fail-fast is out of scope for a bug-fix PR. |
| F8 | `current_molecule_name_from_header` reduces multiple `@molecule` blocks to "last seen"; non-trivial multi-molecule files would mis-assign atoms | LOW (documented limitation in load_mdf docstring) | **Defer.** Out of scope. |
| F9 | `_parse_float_mdf` (writer) silently coerces parse failures to `0.0` instead of preserving the original token | LOW | **Defer.** |
| F10 | The Bond schema's `ix/iy/iz` columns have no docstring describing their semantics (sign convention, meaning of the offset) | LOW | **Fix in this PR.** Add a docstring section to `BONDS_DTYPES`. |
| F11 | No round-trip test asserting `parse(text).bonds == parse(write(parse(text))).bonds` for PBC MDFs | MEDIUM | **Fix in this PR.** Added to `tests/test_pbc_topology.py`. |

## 4. Design

### 4.1 The suffix parser

A new private helper in `_mdf_parser.py`:

```python
# Compiled once at module scope, matches %abc#i where a,b,c are each one signed
# digit and i is one or more digits. Anchored as a search; the suffix may
# appear at the end of a token (after the optional /order... but Materials
# Studio always writes %... before /...).
_MDF_PBC_SUFFIX_RE = re.compile(r"%(-?\d)(-?\d)(-?\d)#(\d+)")


def _parse_image_suffix(token: str) -> tuple[str, tuple[int, int, int]]:
    """Split a Materials Studio MDF connection token's PBC suffix.

    Wire format:   NAME[%abc#i][/order]
      where ``a``, ``b``, ``c`` are each a single signed digit shift along the
      corresponding cell axis, and ``i`` is the Materials Studio image index
      (preserved on connections_raw but not on the parsed Bond, since it
      encodes no geometric information).

    Returns ``(bare_token, (ix, iy, iz))``. If no suffix is present, returns
    the original token unchanged and ``(0, 0, 0)``.

    Examples:
      ``cb%100#1``    → ("cb",  (1, 0, 0))
      ``H%0-10#1``    → ("H",   (0, -1, 0))
      ``Al1%-1-10#5`` → ("Al1", (-1, -1, 0))
      ``XXXX_825:C%0-10#1`` → ("XXXX_825:C", (0, -1, 0))
      ``H``            → ("H",   (0, 0, 0))
    """
```

The order suffix `/order` is stripped independently *before* the image suffix in real Materials Studio output, but in our parser we strip them in either order because the `%abc#i` regex is unambiguous.

### 4.2 Propagating the offset

`build_bonds_from_connections` calls `_parse_image_suffix` *first*, then `parse_target_token` resolves the partner identity. Each emitted Bond row carries `ix`, `iy`, `iz`. The existing `USM.__post_init__` handles `a1 ↔ a2` swap (and the corresponding sign flip) for free.

**Dedup key**: changes from `(a1, a2)` to `(a1, a2, ix, iy, iz)`. Two PBC bonds between the same pair but different image shifts (e.g. `A — B%100#1` and `A — B%-100#1`) are distinct bonds and must both be emitted.

**Reciprocal tokens**: when both `A` lists `B%100#1` and `B` lists `A%-100#1`, the canonicalization (sort by `(a1, a2)` and flip the shift sign on swap) collapses them to the same `(a1=min, a2=max, ix, iy, iz)` triple. The first reciprocal wins; the second is a true duplicate and is correctly deduped.

### 4.3 Writer behavior

`_compose_connections_for_atom` (normalized mode) is updated to:

1. Look up the bond's `(ix, iy, iz)` from `adj_list`.
2. If any non-zero AND the source atom is the canonical `a1` (i.e. `src_aid == a1`), emit `%abc#1` (image index defaults to `1` — Materials Studio's most common value).
3. If `src_aid == a2`, emit the negated shift on the partner's side (so the reciprocal token is correctly signed).

This keeps the round-trip `load(MDF) → save(MDF, normalized=True) → load` semantically faithful even though it doesn't preserve the original `#i`.

### 4.4 Migration impact

- `Bond.ix/iy/iz` were already in `BONDS_DTYPES` with `Int32` dtype and a default of 0. Any consumer that ignored the columns continues to ignore them. Any consumer that previously asserted `all(ix == 0)` on an MDF-loaded structure will see non-zero values on PBC MDFs — **this is the intended behavior change** and the reason for the minor version bump (`2.0.1 → 2.1.0`).
- iff-registry's `topology_instances._extract_pbc_offsets_from_connections` workaround becomes redundant. It can be replaced with a direct read of `structure.bonds[["a1", "a2", "ix", "iy", "iz"]]`. That cleanup is a separate follow-up scope.

## 5. Test plan

### 5.1 Unit tests for `_parse_image_suffix`

Covered in `tests/test_image_suffix_parser.py`:
- All six real-world suffix patterns from the corpus.
- Bare token, no suffix.
- Fully qualified prefix `LABEL_INDEX:NAME` with suffix.
- Token with both `%abc#i` and `/order` in either order.
- Malformed suffix (`%abc` without `#i`, `%XX#1`, etc.) — must fall back to "no suffix" semantics and preserve the literal token.
- Multi-digit image index (`%100#12`).
- Negative on all three axes.

### 5.2 Integration tests in `tests/test_pbc_topology.py`

- A small synthetic MDF (in-memory string passed through a `tempfile`) with a known PBC suffix → assert the resulting Bond has the expected `(ix, iy, iz)`.
- Self-bond with PBC suffix → assert one Bond is emitted with the recovered shift.
- Two bonds between the same pair with different shifts → both emitted, distinct rows.
- Round-trip: `load_mdf → save_mdf(preserve_headers=True) → load_mdf` is byte-equivalent on the connections field (always was) AND on the Bond `(ix, iy, iz)` triples (new).
- Round-trip via normalized writer: `load_mdf → save_mdf(write_normalized_connections=True) → load_mdf` preserves the `(ix, iy, iz)` triples.

### 5.3 Real-data validation (`tests/test_mdf_parser_real_data.py`)

A `pytest.skip()`-guarded fixture loader pulls every `.mdf` from `~/Backups/Dropbox` (only when the directory exists; CI-safe). For each file:
- Extract every `%abc#i` pattern from the raw text via regex.
- Load via USM, count Bonds with non-zero shift.
- Assert: at least as many non-zero-shift bonds as distinct PBC tokens (after reciprocal-dedup).

### 5.4 Regression test for findings F4 and F10

- F4: synthetic single-atom MDF with a self-image bond. Asserts that the Bond is emitted with the recovered shift (instead of being silently dropped as a self-bond).
- F10: assert `BONDS_DTYPES` carries an inline-doc reference to the audit doc.

## 6. Deferred follow-ups

Tracked here for visibility; not in this PR:

- **F3** — preserve image index `i` from `%abc#i`. Would require a new optional `image_index` column on Bond. Justification for deferral: `i` is internal Materials Studio bookkeeping with no geometric meaning, and `connections_raw` is preserved verbatim for any caller that needs it.
- **F5** — emit a structured warning when a connection token doesn't resolve to a known partner. Currently silent.
- **F6** — broaden the `LABEL_INDEX:NAME` prefix grammar.
- **F7** — fail-fast on malformed atom lines instead of silently skipping.
- **F8** — proper per-atom `@molecule` block assignment via interleaved parsing of the atoms section.
- **F9** — preserve original-token text on writer round-trip when float-parse fails.

## 7. Version impact

- **Behavior change**: PBC-laden MDFs now produce Bonds with non-zero `(ix, iy, iz)` instead of all-zero.
- **API change**: none — the columns already existed on the schema and existing consumers continue to read them.
- **Bump**: `2.0.1 → 2.1.0` (minor; behavior fix that surfaces previously-discarded information).
