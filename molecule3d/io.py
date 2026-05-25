"""Readers and writers for molecular coordinate files.

``read`` dispatches on file extension; the individual readers/writers can also
be called directly. PDB parsing uses fixed columns (not whitespace splitting),
which is the only correct way to read the format.
"""

from __future__ import annotations

import os

import numpy as np

from .molecule import Molecule

# altLoc codes we accept: blank (no alternates) or the primary "A" conformation.
_PRIMARY_ALTLOCS = (" ", "A", "")


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
    the PDB spec. Alternate conformations (altLoc) other than the primary one
    are skipped. For multi-model (NMR) files the 1-based ``model`` is returned;
    files without ``MODEL`` records are read in full.
    """
    models = _parse_pdb_models(path)
    if not models:
        return Molecule(np.empty((0, 3)), [], name=_stem(path))
    if not 1 <= model <= len(models):
        raise ValueError(f"model {model} out of range (1..{len(models)})")
    coords, elements = models[model - 1]
    return Molecule(np.array(coords, dtype=float), elements, name=_stem(path))


def read_pdb_models(path: str) -> list[Molecule]:
    """Read every model from a ``.pdb`` file as a list of molecules.

    Single-model files yield a one-element list. Each molecule is named
    ``<stem>#<n>`` so models stay distinguishable.
    """
    stem = _stem(path)
    return [
        Molecule(np.array(coords, dtype=float), elements, name=f"{stem}#{i + 1}")
        for i, (coords, elements) in enumerate(_parse_pdb_models(path))
    ]


def write_xyz(molecule: Molecule, path: str) -> None:
    """Write a molecule to an ``.xyz`` file (unknown elements written as ``X``)."""
    with open(path, "w") as f:
        f.write(f"{len(molecule)}\n{molecule.name}\n")
        for element, (x, y, z) in zip(molecule.elements, molecule.coords):
            f.write(f"{element or 'X':<2} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def write_pdb(molecule: Molecule, path: str) -> None:
    """Write a molecule to a minimal ``.pdb`` file (one residue, chain A)."""
    with open(path, "w") as f:
        for serial, (element, (x, y, z)) in enumerate(
            zip(molecule.elements, molecule.coords), start=1
        ):
            f.write(_pdb_atom_line(serial, element, x, y, z))
        f.write("END\n")


# -- internals --------------------------------------------------------------


def _parse_pdb_models(path: str) -> list[tuple[list, list]]:
    """Return a list of ``(coords, elements)`` tuples, one per model."""
    models: list[tuple[list, list]] = []
    coords: list = []
    elements: list = []

    def flush():
        nonlocal coords, elements
        if coords:
            models.append((coords, elements))
            coords, elements = [], []

    with open(path) as f:
        for line in f:
            record = line[:6].strip()
            if record == "MODEL":
                flush()
            elif record == "ENDMDL":
                flush()
            elif record in ("ATOM", "HETATM"):
                altloc = line[16] if len(line) > 16 else " "
                if altloc not in _PRIMARY_ALTLOCS:
                    continue
                coords.append((
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ))
                element = line[76:78].strip()
                if not element:
                    # Fall back to the atom-name column for older files.
                    element = line[12:16].strip().lstrip("0123456789")[:2]
                elements.append(element)
    flush()
    return models


def _pdb_atom_line(serial: int, element: str, x: float, y: float, z: float) -> str:
    """Build a single fixed-column ``ATOM`` record (80 columns)."""
    line = list(" " * 80)

    def put(value: str, start: int):  # start is 1-based
        line[start - 1:start - 1 + len(value)] = value

    name = (element or "X")[:4]
    put("ATOM", 1)
    put(f"{serial:>5}", 7)
    put(f"{name:<4}", 13)
    put("MOL", 18)
    put("A", 22)
    put(f"{1:>4}", 23)
    put(f"{x:8.3f}", 31)
    put(f"{y:8.3f}", 39)
    put(f"{z:8.3f}", 47)
    put(f"{1.0:6.2f}", 55)
    put(f"{0.0:6.2f}", 61)
    put(f"{(element or 'X'):>2}", 77)
    return "".join(line).rstrip() + "\n"


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
