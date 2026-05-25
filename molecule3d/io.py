"""Readers and writers for molecular coordinate files.

``read`` dispatches on file extension; the individual readers/writers can also
be called directly. PDB parsing uses fixed columns (not whitespace splitting),
which is the only correct way to read the format. Per-atom metadata (atom name,
residue, chain) is captured where the format provides it.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from typing import Optional
from urllib.request import urlopen

import numpy as np

from .molecule import Molecule

# altLoc codes we accept: blank (no alternates) or the primary "A" conformation.
_PRIMARY_ALTLOCS = (" ", "A", "")


def read(path: str) -> Molecule:
    """Read a molecule, picking the parser from the file extension.

    Transparently handles gzip-compressed files (``.pdb.gz``, ``.xyz.gz``).
    """
    ext = _data_extension(path)
    if ext == ".pdb":
        return read_pdb(path)
    if ext == ".xyz":
        return read_xyz(path)
    if ext in (".cif", ".mmcif"):
        return read_cif(path)
    if ext in (".sdf", ".mol"):
        return read_sdf(path)
    raise ValueError(f"Unsupported file type {ext!r}; expected .pdb/.xyz/.cif/.sdf")


def fetch(pdb_id: str, fmt: str = "pdb", cache_dir: Optional[str] = None) -> Molecule:
    """Download a structure from RCSB by its PDB id and read it.

    ``fmt`` is ``"pdb"`` or ``"cif"``. Files are cached (default: the system temp
    directory) so repeat calls don't re-download. Example: ``m3d.fetch("1fqy")``.
    """
    fmt = fmt.lower()
    if fmt not in ("pdb", "cif"):
        raise ValueError("fmt must be 'pdb' or 'cif'")
    pdb_id = pdb_id.lower()
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "molecule3d_cache")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"{pdb_id}.{fmt}")
    if not os.path.exists(dest):
        with urlopen(f"https://files.rcsb.org/download/{pdb_id}.{fmt}") as resp:
            data = resp.read()
        with open(dest, "wb") as fh:
            fh.write(data)
    return read(dest)


def read_xyz(path: str) -> Molecule:
    """Read a single-frame ``.xyz`` file.

    Handles both the standard ``element x y z`` layout and the bare
    ``x y z`` coordinate dumps (with ``#`` comment lines) used by some tools.
    """
    frames = read_xyz_frames(path)
    if not frames:
        return Molecule(np.empty((0, 3)), [], name=_stem(path))
    return frames[0]


def read_xyz_frames(path: str) -> list[Molecule]:
    """Read every frame of a (possibly multi-frame) ``.xyz`` trajectory.

    Standard xyz frames begin with an atom-count line; bare coordinate dumps
    with ``#`` comments are returned as a single frame.
    """
    with _open(path) as f:
        lines = f.readlines()

    stem = _stem(path)
    frames: list[Molecule] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        tokens = lines[i].split()
        if tokens and tokens[0].isdigit():
            count = int(tokens[0])
            block = lines[i + 2:i + 2 + count]
            frames.append(_xyz_block(block, name=f"{stem}#{len(frames) + 1}"))
            i += 2 + count
        else:
            # Bare coordinate dump (no header): consume the rest as one frame.
            frames.append(_xyz_block(lines[i:], name=stem))
            break
    return frames


def read_pdb(path: str, model: int = 1) -> Molecule:
    """Read ``ATOM``/``HETATM`` records from a ``.pdb`` file.

    Coordinates, element, atom name, residue name/id and chain are sliced from
    their fixed columns per the PDB spec. Alternate conformations (altLoc) other
    than the primary one are skipped. For multi-model (NMR) files the 1-based
    ``model`` is returned; files without ``MODEL`` records are read in full.
    """
    models = _parse_pdb_models(path)
    if not models:
        return Molecule(np.empty((0, 3)), [], name=_stem(path))
    if not 1 <= model <= len(models):
        raise ValueError(f"model {model} out of range (1..{len(models)})")
    return _molecule_from_record(models[model - 1], _stem(path))


def read_pdb_models(path: str) -> list[Molecule]:
    """Read every model from a ``.pdb`` file as a list of molecules."""
    stem = _stem(path)
    return [
        _molecule_from_record(rec, f"{stem}#{i + 1}")
        for i, rec in enumerate(_parse_pdb_models(path))
    ]


def read_cif(path: str) -> Molecule:
    """Basic mmCIF reader for standard ``_atom_site`` coordinate loops.

    This parser handles simple whitespace-separated atom-site rows. It is not a
    full mmCIF syntax implementation for quoted values, multiline fields, or
    complex loop constructs.
    """
    columns: list[str] = []
    rows: list[list[str]] = []
    in_atom_site = False
    with _open(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("_atom_site."):
                columns.append(line.split(".", 1)[1])
                in_atom_site = True
                continue
            if in_atom_site:
                if not line or line.startswith(("_", "#", "loop_")):
                    break  # end of the data block
                rows.append(line.split())

    if not columns or not rows:
        raise ValueError(f"no _atom_site records found in {path}")
    idx = {name: i for i, name in enumerate(columns)}

    def col(row, *names, default=""):
        for nm in names:
            if nm in idx and idx[nm] < len(row):
                return row[idx[nm]]
        return default

    coords, els, anames, rnames, rids, chains = [], [], [], [], [], []
    for row in rows:
        coords.append((
            float(row[idx["Cartn_x"]]),
            float(row[idx["Cartn_y"]]),
            float(row[idx["Cartn_z"]]),
        ))
        els.append(col(row, "type_symbol"))
        anames.append(col(row, "label_atom_id", "auth_atom_id"))
        rnames.append(col(row, "label_comp_id", "auth_comp_id"))
        chains.append(col(row, "auth_asym_id", "label_asym_id"))
        rid = col(row, "auth_seq_id", "label_seq_id", default="0")
        rids.append(int(rid) if rid.lstrip("-").isdigit() else 0)

    return Molecule(
        np.array(coords, dtype=float), els, name=_stem(path),
        atom_names=anames, resnames=rnames, resids=np.array(rids, dtype=int),
        chains=chains,
    )


def read_sdf(path: str) -> Molecule:
    """Read the first molecule from an SDF / MDL MOL (V2000) file."""
    with _open(path) as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: too short to be a MOL file")
    counts = lines[3]
    n_atoms = int(counts[:3])
    coords, els = [], []
    for line in lines[4:4 + n_atoms]:
        coords.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
        els.append(line[31:34].strip())
    return Molecule(np.array(coords, dtype=float), els, name=_stem(path))


def write_xyz(molecule: Molecule, path: str) -> None:
    """Write a molecule to an ``.xyz`` file (unknown elements written as ``X``)."""
    with _open(path, "w") as f:
        f.write(f"{len(molecule)}\n{molecule.name}\n")
        for element, (x, y, z) in zip(molecule.elements, molecule.coords):
            f.write(f"{element or 'X':<2} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def write_pdb(molecule: Molecule, path: str) -> None:
    """Write a molecule to a ``.pdb`` file, preserving metadata when present."""
    with _open(path, "w") as f:
        f.write(_molecule_to_pdb_string(molecule))


def _molecule_to_pdb_string(molecule: Molecule) -> str:
    """Serialise a molecule to PDB text, preserving metadata when present."""
    n = len(molecule)
    names = molecule.atom_names or [e or "X" for e in molecule.elements]
    resnames = molecule.resnames or ["MOL"] * n
    chains = molecule.chains or ["A"] * n
    resids = molecule.resids if len(molecule.resids) else np.ones(n, dtype=int)
    lines = [
        _pdb_atom_line(
            serial + 1, names[serial], resnames[serial], chains[serial] or "A",
            int(resids[serial]), molecule.elements[serial], *molecule.coords[serial],
        )
        for serial in range(n)
    ]
    lines.append("END\n")
    return "".join(lines)


# -- internals --------------------------------------------------------------


def _xyz_block(block: list[str], name: str) -> Molecule:
    coords, elements = [], []
    for line in block:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 4 and not _is_float(tokens[0]):
            elements.append(tokens[0])
            coords.append(tuple(float(t) for t in tokens[1:4]))
        else:
            elements.append("")
            coords.append(tuple(float(t) for t in tokens[:3]))
    return Molecule(np.array(coords, dtype=float), elements, name=name)


def _parse_pdb_models(path: str) -> list[dict]:
    """Return a list of per-model records (dict of parallel atom arrays)."""
    models: list[dict] = []
    cur = _new_record()

    def flush():
        nonlocal cur
        if cur["coords"]:
            models.append(cur)
            cur = _new_record()

    with _open(path) as f:
        for line in f:
            record = line[:6].strip()
            if record in ("MODEL", "ENDMDL"):
                flush()
            elif record in ("ATOM", "HETATM"):
                altloc = line[16] if len(line) > 16 else " "
                if altloc not in _PRIMARY_ALTLOCS:
                    continue
                cur["coords"].append((
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ))
                element = line[76:78].strip()
                if not element:
                    element = line[12:16].strip().lstrip("0123456789")[:2]
                cur["elements"].append(element)
                cur["atom_names"].append(line[12:16].strip())
                cur["resnames"].append(line[17:20].strip())
                cur["chains"].append(line[21].strip())
                resid = line[22:26].strip()
                cur["resids"].append(int(resid) if resid.lstrip("-").isdigit() else 0)
    flush()
    return models


def _new_record() -> dict:
    return {k: [] for k in ("coords", "elements", "atom_names", "resnames",
                            "chains", "resids")}


def _molecule_from_record(rec: dict, name: str) -> Molecule:
    return Molecule(
        np.array(rec["coords"], dtype=float), rec["elements"], name=name,
        atom_names=rec["atom_names"], resnames=rec["resnames"],
        resids=np.array(rec["resids"], dtype=int), chains=rec["chains"],
    )


def _pdb_atom_line(serial, atom_name, resname, chain, resid, element, x, y, z):
    """Build a single fixed-column ``ATOM`` record (80 columns)."""
    line = list(" " * 80)

    def put(value: str, start: int):  # start is 1-based
        line[start - 1:start - 1 + len(value)] = value

    put("ATOM", 1)
    put(f"{serial:>5}", 7)
    put(f"{(atom_name or 'X')[:4]:<4}", 13)
    put(f"{(resname or 'MOL')[:3]:>3}", 18)
    put((chain or "A")[0], 22)
    put(f"{resid:>4}", 23)
    put(f"{x:8.3f}", 31)
    put(f"{y:8.3f}", 39)
    put(f"{z:8.3f}", 47)
    put(f"{1.0:6.2f}", 55)
    put(f"{0.0:6.2f}", 61)
    put(f"{(element or 'X'):>2}", 77)
    return "".join(line).rstrip() + "\n"


def _open(path: str, mode: str = "r"):
    """Open a file, transparently handling gzip by the ``.gz`` suffix."""
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t")
    return open(path, mode)


def _data_extension(path: str) -> str:
    """Extension ignoring a trailing ``.gz`` (so ``a.pdb.gz`` -> ``.pdb``)."""
    base = path[:-3] if path.endswith(".gz") else path
    return os.path.splitext(base)[1].lower()


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _stem(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = base[:-3]
    return os.path.splitext(base)[0]
