"""Readers for molecular coordinate files.

``read`` dispatches on file extension; the individual readers can also be
called directly. PDB parsing uses fixed columns (not whitespace splitting),
which is the only correct way to read the format.
"""

from __future__ import annotations

import os

import numpy as np

from .molecule import Molecule


def read(path: str) -> Molecule:
    """Read a molecule, picking the parser from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdb":
        return read_pdb(path)
    if ext == ".xyz":
        return read_xyz(path)
    raise ValueError(f"Unsupported file type {ext!r}; expected .pdb or .xyz")


def read_xyz(path: str) -> Molecule:
    """Read an ``.xyz`` file.

    Handles both the standard ``element x y z`` layout and the bare
    ``x y z`` coordinate dumps (with ``#`` comment lines) used by some tools.
    A leading integer atom-count line and a following comment line, if present,
    are skipped.
    """
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    with open(path) as f:
        lines = f.readlines()

    start = 0
    if lines and lines[0].split()[:1] and lines[0].split()[0].isdigit():
        # Standard xyz header: count line + comment line.
        start = 2

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 4 and not _is_float(tokens[0]):
            element = tokens[0]
            x, y, z = (float(t) for t in tokens[1:4])
        else:
            element = ""
            x, y, z = (float(t) for t in tokens[:3])
        coords.append((x, y, z))
        elements.append(element)

    return Molecule(np.array(coords, dtype=float), elements, name=_stem(path))


def read_pdb(path: str, model: int = 1) -> Molecule:
    """Read ``ATOM``/``HETATM`` records from a ``.pdb`` file.

    Coordinates and the element symbol are sliced from their fixed columns per
    the PDB spec. For multi-model (NMR) files only the requested ``model`` is
    returned; files without ``MODEL`` records are read in full.
    """
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    current_model = 0
    has_models = False

    with open(path) as f:
        for line in f:
            record = line[:6].strip()
            if record == "MODEL":
                has_models = True
                current_model = int(line[10:14])
            elif record == "ENDMDL":
                if has_models and current_model == model:
                    break
            elif record in ("ATOM", "HETATM"):
                if has_models and current_model != model:
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip()
                if not element:
                    # Fall back to the atom-name column for older files.
                    element = line[12:16].strip().lstrip("0123456789")[:2]
                coords.append((x, y, z))
                elements.append(element)

    return Molecule(np.array(coords, dtype=float), elements, name=_stem(path))


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
