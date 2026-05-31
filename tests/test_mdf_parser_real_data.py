"""Real-data validation for the MDF parser PBC suffix fix.

This test pulls every ``.mdf`` from Sean's local Dropbox corpus
(``~/Backups/Dropbox/**/*.mdf``) and asserts that for each file containing
Materials Studio ``%abc#i`` suffixes, the parsed ``USM.bonds`` table
recovers the expected non-zero ``(ix, iy, iz)`` triples.

The corpus is local-only; CI / fresh checkouts won't have it, so the
fixture loader uses ``pytest.skip()`` when the directory is absent.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import pytest

from usm.io.mdf import load_mdf


_CORPUS_ROOT = Path(os.path.expanduser("~/Backups/Dropbox"))
_MDF_PBC_RE = re.compile(r"%(-?\d)(-?\d)(-?\d)#(\d+)")


def _discover_pbc_mdfs(root: Path, limit: int = 10) -> list[Path]:
    """Return up to ``limit`` MDFs from the corpus that contain at least one
    Materials Studio ``%abc#i`` suffix. Walks lazily so we don't read the
    whole corpus when only a handful of samples are needed."""
    if not root.exists():
        return []
    sampled: list[Path] = []
    for p in root.rglob("*.mdf"):
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                head = f.read(200_000)
        except OSError:
            continue
        if _MDF_PBC_RE.search(head):
            sampled.append(p)
            if len(sampled) >= limit:
                break
    return sampled


def _expected_shifts_from_raw(text: str) -> set[tuple[int, int, int]]:
    """Distinct non-zero shifts that appear in raw MDF text."""
    out: set[tuple[int, int, int]] = set()
    for m in _MDF_PBC_RE.finditer(text):
        s = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if s != (0, 0, 0):
            out.add(s)
    return out


@pytest.fixture(scope="module")
def _pbc_mdf_samples() -> Iterable[Path]:
    samples = _discover_pbc_mdfs(_CORPUS_ROOT, limit=10)
    if not samples:
        pytest.skip(
            f"No PBC-laden MDFs found under {_CORPUS_ROOT}; "
            "skipping real-data validation."
        )
    return samples


def test_real_mdfs_recover_pbc_shifts(_pbc_mdf_samples):
    """For every sampled real MDF, every distinct non-zero ``%abc#i`` shift
    that appears in the raw text shows up on at least one parsed Bond.

    This is the contractual claim of the fix: no PBC information visible in
    the raw MDF should be silently lost after USM parsing.
    """
    failures: list[str] = []
    for path in _pbc_mdf_samples:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        expected = _expected_shifts_from_raw(raw)

        usm = load_mdf(str(path))
        if usm.bonds is None or len(usm.bonds) == 0:
            failures.append(f"{path.name}: no bonds parsed (expected shifts: {expected})")
            continue

        # Build the set of {|shift|} that appear on parsed bonds. We use the
        # canonical shift directly (USM already normalized a1 <= a2 and
        # flipped the sign on swap). To compare against the raw-text shifts,
        # we accept either the shift or its negation, since one side of the
        # reciprocal pair will canonicalize to ``shift`` and the other to
        # ``-shift`` depending on aid ordering.
        observed: set[tuple[int, int, int]] = set()
        for _, r in usm.bonds.iterrows():
            s = (int(r["ix"]), int(r["iy"]), int(r["iz"]))
            if s != (0, 0, 0):
                observed.add(s)
                observed.add((-s[0], -s[1], -s[2]))

        missing = {s for s in expected if s not in observed}
        if missing:
            failures.append(
                f"{path.name}: missing shifts {missing}; "
                f"observed (with negations) {len(observed)} distinct."
            )

    assert not failures, (
        "Real-data PBC validation failures:\n  " + "\n  ".join(failures)
    )


def test_real_mdfs_bonds_count_at_least_distinct_shifts(_pbc_mdf_samples):
    """A weaker invariant: number of bonds with non-zero shift is at least
    the number of distinct non-zero shifts in the raw text. Catches the
    "all zeroes" regression even if a single representative bond carries
    every shift (which shouldn't happen but would still pass the previous
    assertion)."""
    for path in _pbc_mdf_samples:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        expected = _expected_shifts_from_raw(raw)
        if not expected:
            continue
        usm = load_mdf(str(path))
        if usm.bonds is None:
            pytest.fail(f"{path.name}: no bonds")
        nonzero = sum(
            1
            for _, r in usm.bonds.iterrows()
            if (int(r["ix"]), int(r["iy"]), int(r["iz"])) != (0, 0, 0)
        )
        assert nonzero >= len(expected), (
            f"{path.name}: only {nonzero} non-zero-shift bonds vs {len(expected)} "
            f"distinct shifts in raw text"
        )
