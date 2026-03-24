"""Unified file loader with format auto-detection by extension.

Usage:
    from usm.io.loader import load
    usm = load("structure.cif")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from usm.core.model import USM

_EXTENSION_MAP: dict[str, str] = {
    ".car": "car",
    ".mdf": "mdf",
    ".cif": "cif",
    ".pdb": "pdb",
}

_SUPPORTED = sorted(_EXTENSION_MAP.keys())


def load(path: str, **kwargs: Any) -> USM:
    """Load a structure file, auto-detecting format from extension.

    Supported extensions: .car, .mdf, .cif, .pdb

    Args:
        path: Path to structure file.
        **kwargs: Passed to the format-specific loader.

    Returns:
        USM structure.

    Raises:
        ValueError: If the file extension is not recognized.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    fmt = _EXTENSION_MAP.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {_SUPPORTED}"
        )

    if fmt == "car":
        from usm.io.car import load_car
        return load_car(str(p), **kwargs)
    elif fmt == "mdf":
        from usm.io.mdf import load_mdf
        return load_mdf(str(p), **kwargs)
    elif fmt == "cif":
        from usm.io.cif import load_cif
        return load_cif(str(p), **kwargs)
    elif fmt == "pdb":
        from usm.io.pdb import load_pdb
        return load_pdb(str(p), **kwargs)

    raise ValueError(f"Format '{fmt}' not implemented")


__all__ = ["load"]
