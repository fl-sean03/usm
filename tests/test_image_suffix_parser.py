"""Unit tests for ``_parse_image_suffix`` — the Materials Studio PBC suffix parser.

Wire format the parser must handle (one token, whitespace-delimited)::

    NAME[%abc#i][/order]

where ``a``, ``b``, ``c`` are each a single signed digit shift along the
corresponding cell axis and ``i`` is the Materials Studio image index
(positive integer; not preserved on the parsed Bond — see audit doc F3).

These tests cover the grammar exhaustively against the six real-world
suffix patterns observed in Sean's MDF corpus plus the malformed-input
cases the parser should treat as no-suffix.
"""
from __future__ import annotations

import pytest

from usm.io._mdf_parser import _parse_image_suffix


# --- positive cases drawn from the real corpus ---


@pytest.mark.parametrize(
    "token, expected_bare, expected_shift",
    [
        # Six distinct real-world patterns from
        # ~/Backups/Dropbox/**/*.mdf (corpus scan in MDF_PARSER_AUDIT.md §1).
        ("O6%110#3", "O6", (1, 1, 0)),
        ("O4%010#3", "O4", (0, 1, 0)),
        ("O2%110#3", "O2", (1, 1, 0)),
        ("O5%0-10#1", "O5", (0, -1, 0)),
        ("O3%100#3", "O3", (1, 0, 0)),
        ("Al1%-1-10#5", "Al1", (-1, -1, 0)),
        # From the task brief
        ("cb%100#1", "cb", (1, 0, 0)),
        ("H%0-10#1", "H", (0, -1, 0)),
        # Fully-qualified target with suffix
        ("XXXX_825:C%0-10#1", "XXXX_825:C", (0, -1, 0)),
        ("XXXX_1:Al1%110#3", "XXXX_1:Al1", (1, 1, 0)),
    ],
)
def test_real_world_suffixes(token, expected_bare, expected_shift):
    bare, shift = _parse_image_suffix(token)
    assert bare == expected_bare
    assert shift == expected_shift


# --- absent / malformed suffix → no-op ---


@pytest.mark.parametrize(
    "token",
    [
        "H",
        "C1",
        "XXXX_1:Al1",
        "XXXX_1729:H",
        "",
    ],
)
def test_no_suffix_is_identity(token):
    bare, shift = _parse_image_suffix(token)
    assert bare == token
    assert shift == (0, 0, 0)


@pytest.mark.parametrize(
    "token",
    [
        # Missing image index (no `#i`)
        "cb%100",
        # Non-digit shift
        "cb%abc#1",
        # Only two axes
        "cb%10#1",
        # Lone %
        "cb%",
        # % without trailing axes at all
        "cb%#1",
    ],
)
def test_malformed_suffix_is_treated_as_no_suffix(token):
    """A malformed suffix must not crash and must not pretend it parsed."""
    bare, shift = _parse_image_suffix(token)
    # The parser doesn't fall through to a "best-effort" parse — it
    # treats anything that doesn't match the strict regex as no suffix.
    # The token comes back unchanged.
    assert bare == token
    assert shift == (0, 0, 0)


# --- order suffix interaction ---


def test_order_suffix_preserved_after_image_strip():
    """``%abc#i`` is removed, but ``/order`` (anywhere in the token) is left
    alone for the caller's slash-split logic to handle."""
    bare, shift = _parse_image_suffix("Al1%-1-10#5/1.5")
    assert bare == "Al1/1.5"
    assert shift == (-1, -1, 0)


def test_order_only_no_image():
    bare, shift = _parse_image_suffix("H/2.0")
    assert bare == "H/2.0"
    assert shift == (0, 0, 0)


# --- multi-digit image index ---


def test_multi_digit_image_index():
    """``#i`` may be multiple digits; the geometric info is the same."""
    bare, shift = _parse_image_suffix("C%100#12")
    assert bare == "C"
    assert shift == (1, 0, 0)


def test_image_index_does_not_leak_into_shift():
    """A multi-digit index must not be mistaken for a shift component."""
    bare, shift = _parse_image_suffix("C%100#100")
    assert bare == "C"
    assert shift == (1, 0, 0)


# --- defensive cases ---


def test_non_string_input():
    bare, shift = _parse_image_suffix(None)  # type: ignore[arg-type]
    assert bare is None
    assert shift == (0, 0, 0)


def test_signed_zero():
    """``-0`` is treated as 0; the regex would match ``-0`` as a digit-with-sign."""
    bare, shift = _parse_image_suffix("X%-0-00#1")
    assert bare == "X"
    assert shift == (0, 0, 0)
