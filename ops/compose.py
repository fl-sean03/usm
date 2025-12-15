from __future__ import annotations

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from usm.core.model import USM


KEYS_DEFAULT = ["mol_label", "mol_index", "name"]


def compose_on_keys(
    primary: USM,
    secondary: USM,
    keys: Optional[List[str]] = None,
    *,
    policy: str = "silent",
    coverage_threshold: float = 0.95,
    return_report: bool = False,
) -> USM | tuple[USM, Dict[str, Any]]:
    """
    Compose two USM structures by key-joining their atoms, filling missing columns in 'primary'
    from 'secondary'. Typical use: combine CAR (coords) with MDF (topology/bonds).

    Parameters
    - keys: columns to join on (default ["mol_label","mol_index","name"])
    - policy: "silent" (default), "warn", or "error_below_coverage"
      - "silent": compute coverage metrics and attach to provenance["compose_coverage"]
      - "warn": as silent, and append a parse_notes warning if coverage_ratio < coverage_threshold
      - "error_below_coverage": raise ValueError if coverage_ratio < coverage_threshold
    - coverage_threshold: float threshold for coverage policy (default 0.95)
    - return_report: if True, returns (USM, report_dict) where report_dict mirrors provenance["compose_coverage"]

    Behavior
    - Bonds: prefer secondary bonds if present; otherwise keep primary bonds.
    - Cell: prefer primary cell unless missing (then use secondary).
    - Provenance/preserved_text: prefer primary, append parse_notes from secondary; attach compose_coverage metrics.
    """
    keys = keys or list(KEYS_DEFAULT)

    a_primary = primary.atoms.copy()
    a_secondary = secondary.atoms.copy()

    # Ensure key columns exist
    for k in keys:
        if k not in a_primary.columns or k not in a_secondary.columns:
            raise ValueError(f"compose_on_keys requires key column '{k}' present in both inputs")

    # Compute composition coverage over UNIQUE key tuples (deterministic last-wins semantics elsewhere)
    primary_key_tuples = list(a_primary[keys].itertuples(index=False, name=None))
    secondary_key_tuples = list(a_secondary[keys].itertuples(index=False, name=None))
    primary_keys_set = set(primary_key_tuples)
    secondary_keys_set = set(secondary_key_tuples)
    matched_set = primary_keys_set.intersection(secondary_keys_set)
    left_only_set = primary_keys_set.difference(secondary_keys_set)
    right_only_set = secondary_keys_set.difference(primary_keys_set)

    matched_count = len(matched_set)
    primary_total = len(primary_keys_set)
    secondary_total = len(secondary_keys_set)
    left_only_count = len(left_only_set)
    right_only_count = len(right_only_set)
    coverage_ratio = matched_count / max(primary_total, 1)

    # Prepare deterministic message and policy handling
    warn_msg = None
    coverage_metrics: Dict[str, Any] = {
        "policy": str(policy),
        "coverage_threshold": float(coverage_threshold),
        "keys": list(keys),
        "matched_count": int(matched_count),
        "left_only_count": int(left_only_count),
        "right_only_count": int(right_only_count),
        "primary_total": int(primary_total),
        "secondary_total": int(secondary_total),
        "coverage_ratio": float(coverage_ratio),
    }
    if coverage_ratio < float(coverage_threshold):
        msg = (
            "compose_on_keys coverage below threshold: "
            f"ratio={coverage_ratio:.6f} threshold={float(coverage_threshold):.6f} "
            f"matched={matched_count} left_only={left_only_count} right_only={right_only_count} "
            f"primary_total={primary_total} secondary_total={secondary_total} "
            f"keys=[{','.join(keys)}]"
        )
        if policy == "warn":
            warn_msg = msg
        elif policy == "error_below_coverage":
            # Raise early with informative message
            raise ValueError(msg)
        # policy == "silent": proceed

    # Prepare columns to bring from secondary that are missing or NA in primary
    sec_only_cols = [c for c in a_secondary.columns if c not in a_primary.columns]
    # For overlapping columns, we only fill NAs in primary from secondary
    overlap_cols = [c for c in a_secondary.columns if c in a_primary.columns and c not in keys and c != "aid"]

    # Build a trimmed secondary table with unique keys (last occurrence wins deterministically)
    sec_trim = a_secondary.sort_values(by=keys).drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    # For overlap columns, rename with suffix for merge, then fill
    overlap_map = {c: f"__sec_{c}" for c in overlap_cols}
    sec_overlap = sec_trim[keys + overlap_cols].rename(columns=overlap_map)

    # Merge overlap fill-ins
    merged = a_primary.merge(sec_overlap, on=keys, how="left")

    # Fill overlapping NA values
    for c in overlap_cols:
        sc = f"__sec_{c}"
        if sc in merged.columns:
            mask = merged[c].isna()
            if mask.any():
                merged.loc[mask, c] = merged.loc[mask, sc]
            merged.drop(columns=[sc], inplace=True)

    # Add secondary-only columns (aligned by keys)
    if sec_only_cols:
        sec_only = sec_trim[keys + sec_only_cols]
        merged = merged.merge(sec_only, on=keys, how="left", suffixes=("", ""))

    # Reassign aids deterministically
    merged = merged.reset_index(drop=True)
    if "aid" in merged.columns:
        merged = merged.drop(columns=["aid"])
    merged.insert(0, "aid", np.arange(len(merged), dtype=np.int32))

    # Choose bonds: prefer secondary bonds (e.g., from MDF connections); otherwise primary bonds
    out_bonds = None
    if secondary.bonds is not None and len(secondary.bonds) > 0:
        out_bonds = secondary.bonds.copy()
        # Remap bonds a1/a2 to new aids based on keys mapping
        # Build key -> new aid index for the merged atoms
        key_tuples = merged[keys].itertuples(index=False, name=None)
        new_aids = merged["aid"].astype(int).to_numpy()
        key_to_aid: Dict[tuple, int] = {tuple(k): int(a) for k, a in zip(key_tuples, new_aids)}

        # Build mapping for secondary atoms' aids using their keys
        sec_keys = secondary.atoms[keys].itertuples(index=False, name=None)
        sec_aids = secondary.atoms["aid"].astype(int).to_numpy()
        sec_key_to_aid: Dict[tuple, int] = {tuple(k): int(a) for k, a in zip(sec_keys, sec_aids)}

        # Remap by translating secondary aid -> key -> new aid
        def _remap_series(aid_series: pd.Series) -> pd.Series:
            # Convert each secondary aid to its key then to new aid
            vals = []
            idx_map = {v: k for k, v in sec_key_to_aid.items()}
            for v in aid_series.astype(int).tolist():
                k = idx_map.get(int(v))
                if k is None:
                    vals.append(pd.NA)
                else:
                    vals.append(key_to_aid.get(k, pd.NA))
            return pd.Series(vals, index=aid_series.index)

        out_bonds["a1"] = _remap_series(out_bonds["a1"])
        out_bonds["a2"] = _remap_series(out_bonds["a2"])
        out_bonds = out_bonds.dropna(subset=["a1", "a2"]).copy()
        out_bonds["a1"] = out_bonds["a1"].astype("int32")
        out_bonds["a2"] = out_bonds["a2"].astype("int32")
        # Normalize a1 < a2
        a1v = out_bonds["a1"].to_numpy()
        a2v = out_bonds["a2"].to_numpy()
        swap = a1v > a2v
        if swap.any():
            tmp = a1v[swap].copy()
            a1v[swap] = a2v[swap]
            a2v[swap] = tmp
        out_bonds["a1"] = a1v
        out_bonds["a2"] = a2v
        out_bonds = out_bonds.reset_index(drop=True)
    elif primary.bonds is not None and len(primary.bonds) > 0:
        out_bonds = primary.bonds.copy()

    # Choose cell
    out_cell = dict(primary.cell or {})
    if not out_cell:
        out_cell = dict(secondary.cell or {})

    # Merge provenance and preserved_text: prefer primary, append note, attach coverage metrics and optional warning
    provenance = dict(primary.provenance or {})
    pnotes = provenance.get("parse_notes", "")
    append_note = secondary.provenance.get("source_path", None) if secondary.provenance else None
    if append_note:
        pnotes = (pnotes + " | " if pnotes else "") + f"composed with {append_note}"
    if warn_msg:
        pnotes = (pnotes + " | " if pnotes else "") + warn_msg
    if pnotes:
        provenance["parse_notes"] = pnotes
    # Attach structured coverage metrics
    provenance["compose_coverage"] = coverage_metrics

    preserved_text = dict(primary.preserved_text or {})

    out_usm = USM(
        atoms=merged,
        bonds=out_bonds,
        molecules=None,  # optional in v0.1
        cell=out_cell,
        provenance=provenance,
        preserved_text=preserved_text,
    )
    if return_report:
        return out_usm, coverage_metrics
    return out_usm