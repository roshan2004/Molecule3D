"""Readers and writers for molecular coordinate files.

``read`` dispatches on file extension; the individual readers/writers can also
be called directly. PDB parsing uses fixed columns (not whitespace splitting),
which is the only correct way to read the format. Per-atom metadata (atom name,
residue, chain) and explicit connectivity are captured where the format
provides them.
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
_ALTLOC_POLICIES = ("primary", "first", "highest_occupancy", "all")


def read(path: str) -> Molecule:
    """Read a molecule, picking the parser from the file extension.

    Transparently handles gzip-compressed files (``.pdb.gz``, ``.xyz.gz``).
    """
    path = os.fspath(path)
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
    directory) so repeat calls don't re-download. Example: ``ms.fetch("1fqy")``.
    """
    fmt = fmt.lower()
    if fmt not in ("pdb", "cif"):
        raise ValueError("fmt must be 'pdb' or 'cif'")
    pdb_id = pdb_id.lower()
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "molscope_cache")
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


def read_pdb(path: str, model: int = 1, altloc: str = "primary") -> Molecule:
    """Read ``ATOM``/``HETATM`` records from a ``.pdb`` file.

    Coordinates, element, atom name, residue name/id and chain are sliced from
    their fixed columns per the PDB spec. ``altloc`` controls alternate
    conformations: ``"primary"`` keeps blank/``A`` locations, ``"first"`` keeps
    the first location per atom, ``"highest_occupancy"`` keeps the location with
    the largest occupancy, and ``"all"`` keeps every alternate. For multi-model
    (NMR) files the 1-based ``model`` is returned; files without ``MODEL``
    records are read in full.
    """
    models = _parse_pdb_models(path, altloc=altloc)
    if not models:
        return Molecule(np.empty((0, 3)), [], name=_stem(path))
    if not 1 <= model <= len(models):
        raise ValueError(f"model {model} out of range (1..{len(models)})")
    return _molecule_from_record(models[model - 1], _stem(path))


def read_pdb_models(path: str, altloc: str = "primary") -> list[Molecule]:
    """Read every model from a ``.pdb`` file as a list of molecules."""
    stem = _stem(path)
    return [
        _molecule_from_record(rec, f"{stem}#{i + 1}")
        for i, rec in enumerate(_parse_pdb_models(path, altloc=altloc))
    ]


def read_cif(path: str, parser: str = "builtin") -> Molecule:
    """Basic mmCIF reader for standard ``_atom_site`` coordinate loops.

    ``parser="builtin"`` uses MolScope's lightweight atom-site reader.
    ``parser="gemmi"`` uses the optional Gemmi backend
    (``pip install "molscope[cif]"``). Neither parser performs full dictionary
    validation; use :func:`molscope.validate_cif` for validation checks.
    """
    path = os.fspath(path)
    if parser == "gemmi":
        return _read_cif_gemmi(path)
    if parser != "builtin":
        raise ValueError("parser must be 'builtin' or 'gemmi'")
    return _read_cif_builtin(path)


def _read_cif_builtin(path: str) -> Molecule:
    """Read a CIF/mmCIF atom-site loop with MolScope's lightweight tokenizer."""
    with _open(path) as f:
        tokens = _cif_tokens(f.read())

    columns: list[str] = []
    rows: list[list[str]] = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() != "loop_":
            i += 1
            continue

        i += 1
        loop_columns = []
        while i < len(tokens) and tokens[i].startswith("_"):
            loop_columns.append(tokens[i])
            i += 1

        if not any(col.lower().startswith("_atom_site.") for col in loop_columns):
            while i < len(tokens) and not _cif_control_token(tokens[i]):
                i += 1
            continue

        width = len(loop_columns)
        columns = [col.split(".", 1)[1] if "." in col else col for col in loop_columns]
        while i + width <= len(tokens) and not _cif_control_token(tokens[i]):
            row = tokens[i:i + width]
            if any(_cif_control_token(tok) for tok in row):
                break
            rows.append(row)
            i += width
        break

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


def _read_cif_gemmi(path: str) -> Molecule:
    """Read a CIF/mmCIF atom-site loop with Gemmi when the optional extra exists."""
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "Gemmi is required for parser='gemmi'; install it with "
            'pip install "molscope[cif]"'
        ) from exc

    doc = gemmi.cif.read_file(path)
    for block in doc:
        col = block.find_loop("_atom_site.Cartn_x")
        if not col:
            continue
        loop = col.get_loop()
        tags = [tag.split(".", 1)[1] if "." in tag else tag for tag in loop.tags]
        rows = _gemmi_loop_rows(loop)
        return _molecule_from_cif_rows(tags, rows, _stem(path))
    raise ValueError(f"no _atom_site records found in {path}")


def _molecule_from_cif_rows(columns: list[str], rows: list[list[str]], name: str) -> Molecule:
    idx = {col: i for i, col in enumerate(columns)}

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
        np.array(coords, dtype=float), els, name=name,
        atom_names=anames, resnames=rnames, resids=np.array(rids, dtype=int),
        chains=chains,
    )


def read_sdf(path: str) -> Molecule:
    """Read the first molecule from an SDF / MDL MOL (V2000) file.

    Atom coordinates, element symbols, explicit bonds and V2000 bond orders are
    preserved.
    """
    with _open(path) as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: too short to be a MOL file")
    counts = lines[3]
    n_atoms = int(counts[:3])
    n_bonds = int(counts[3:6])
    coords, els, formal_charges = [], [], []
    for line in lines[4:4 + n_atoms]:
        coords.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
        els.append(line[31:34].strip())
        formal_charges.append(_sdf_charge_from_code(line[36:39].strip()))

    bonds, orders = [], []
    for line in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
        a = int(line[0:3]) - 1
        b = int(line[3:6]) - 1
        order = int(line[6:9])
        bonds.append((a, b))
        orders.append(_sdf_bond_order(order))

    for line in lines[4 + n_atoms + n_bonds:]:
        if line.startswith("M  END"):
            break
        if line.startswith("M  CHG"):
            _apply_sdf_charge_line(line, formal_charges)

    bond_index = np.array(bonds, dtype=int).reshape(-1, 2)
    bond_orders = np.array(orders, dtype=float) if bonds else None
    return Molecule(
        np.array(coords, dtype=float), els, name=_stem(path),
        bond_index=bond_index if bonds else None,
        bond_orders=bond_orders,
        formal_charges=np.array(formal_charges, dtype=int),
    )


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
    if molecule.bond_index is not None:
        lines.extend(
            _pdb_conect_line(int(i) + 1, int(j) + 1)
            for i, j in molecule.bond_index
        )
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


def _cif_tokens(text: str) -> list[str]:
    """Tokenize enough CIF/mmCIF syntax for atom-site coordinate loops."""
    tokens: list[str] = []
    i, n = 0, len(text)
    at_line_start = True
    while i < n:
        char = text[i]
        if at_line_start and char == ";":
            i += 1
            start = i
            end = text.find("\n;", i)
            if end == -1:
                tokens.append(text[start:].rstrip("\n"))
                break
            tokens.append(text[start:end])
            i = end + 2
            while i < n and text[i] != "\n":
                i += 1
            at_line_start = True
            continue
        if char.isspace():
            at_line_start = char == "\n"
            i += 1
            continue
        if char == "#":
            while i < n and text[i] != "\n":
                i += 1
            at_line_start = True
            continue
        if char in ("'", '"'):
            quote = char
            i += 1
            start = i
            while i < n and text[i] != quote:
                i += 1
            tokens.append(text[start:i])
            i += 1 if i < n else 0
            at_line_start = False
            continue

        start = i
        while i < n and not text[i].isspace() and text[i] != "#":
            i += 1
        tokens.append(text[start:i])
        at_line_start = False
    return tokens


def _gemmi_loop_rows(loop) -> list[list[str]]:
    width = loop.width()
    values = list(loop.values)
    return [values[i:i + width] for i in range(0, len(values), width)]


def _cif_control_token(token: str) -> bool:
    lower = token.lower()
    return lower == "loop_" or lower.startswith(("data_", "save_")) or token.startswith("_")


def _sdf_bond_order(code: int) -> float:
    # V2000 uses 4 for aromatic bonds, not a numeric quadruple bond.
    return 1.5 if code == 4 else float(code)


def _sdf_charge_from_code(token: str) -> int:
    if not token:
        return 0
    try:
        code = int(token)
    except ValueError:
        return 0
    return {0: 0, 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}.get(code, 0)


def _apply_sdf_charge_line(line: str, charges: list[int]) -> None:
    tokens = line.split()
    if len(tokens) < 4:
        return
    try:
        count = int(tokens[2])
    except ValueError:
        return
    pairs = tokens[3:3 + 2 * count]
    for i in range(0, len(pairs), 2):
        try:
            atom_idx = int(pairs[i]) - 1
            charge = int(pairs[i + 1])
        except (ValueError, IndexError):
            continue
        if 0 <= atom_idx < len(charges):
            charges[atom_idx] = charge


def _parse_pdb_models(path: str, altloc: str = "primary") -> list[dict]:
    """Return a list of per-model records (dict of parallel atom arrays)."""
    _validate_altloc_policy(altloc)
    models: list[dict] = []
    cur = _new_record()
    global_conect: list[tuple[int, int]] = []

    def flush():
        nonlocal cur
        if cur["atoms"]:
            models.append(_record_from_atoms(cur["atoms"], cur["conect"], altloc))
            cur = _new_record()

    with _open(path) as f:
        for line in f:
            record = line[:6].strip()
            if record in ("MODEL", "ENDMDL"):
                flush()
            elif record in ("ATOM", "HETATM"):
                cur["atoms"].append(_parse_pdb_atom(line, len(cur["atoms"]) + 1))
            elif record == "CONECT":
                pairs = _parse_conect(line)
                if cur["atoms"]:
                    cur["conect"].extend(pairs)
                else:
                    global_conect.extend(pairs)
    flush()
    if global_conect:
        for model in models:
            model["conect"].extend(global_conect)
    return models


def _new_record() -> dict:
    return {"atoms": [], "conect": []}


def _validate_altloc_policy(altloc: str) -> None:
    if altloc not in _ALTLOC_POLICIES:
        choices = "', '".join(_ALTLOC_POLICIES)
        raise ValueError(f"altloc must be one of '{choices}', got {altloc!r}")


def _parse_pdb_atom(line: str, fallback_serial: int) -> dict:
    serial = line[6:11].strip()
    serial_value = int(serial) if serial.lstrip("-").isdigit() else fallback_serial
    element = line[76:78].strip()
    atom_name = line[12:16].strip()
    if not element:
        element = atom_name.lstrip("0123456789")[:2]
    resid = line[22:26].strip()
    return {
        "serial": serial_value,
        "coords": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        "element": element,
        "atom_name": atom_name,
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resid": int(resid) if resid.lstrip("-").isdigit() else 0,
        "icode": line[26].strip() if len(line) > 26 else "",
        "altloc": line[16] if len(line) > 16 else " ",
        "occupancy": _pdb_float(line[54:60], default=0.0),
    }


def _record_from_atoms(atoms: list[dict], conect: list[tuple[int, int]], altloc: str) -> dict:
    selected = _select_altloc_atoms(atoms, altloc)
    return {
        "coords": [atom["coords"] for atom in selected],
        "elements": [atom["element"] for atom in selected],
        "atom_names": [atom["atom_name"] for atom in selected],
        "resnames": [atom["resname"] for atom in selected],
        "chains": [atom["chain"] for atom in selected],
        "resids": [atom["resid"] for atom in selected],
        "serials": [atom["serial"] for atom in selected],
        "conect": list(conect),
    }


def _select_altloc_atoms(atoms: list[dict], altloc: str) -> list[dict]:
    if altloc == "all":
        return atoms
    if altloc == "primary":
        return [atom for atom in atoms if atom["altloc"] in _PRIMARY_ALTLOCS]

    selected: dict[tuple, tuple[int, dict]] = {}
    for order, atom in enumerate(atoms):
        key = _altloc_group_key(atom)
        if key not in selected:
            selected[key] = (order, atom)
            continue
        if altloc == "highest_occupancy":
            selected[key] = _best_altloc(selected[key], (order, atom))
    return [atom for _, atom in sorted(selected.values(), key=lambda item: item[0])]


def _altloc_group_key(atom: dict) -> tuple:
    return (
        atom["chain"],
        atom["resid"],
        atom["icode"],
        atom["resname"],
        atom["atom_name"],
    )


def _best_altloc(left: tuple[int, dict], right: tuple[int, dict]) -> tuple[int, dict]:
    _, left_atom = left
    _, right_atom = right
    if right_atom["occupancy"] > left_atom["occupancy"]:
        return right
    if right_atom["occupancy"] < left_atom["occupancy"]:
        return left
    if _altloc_rank(right_atom["altloc"]) < _altloc_rank(left_atom["altloc"]):
        return right
    return left


def _altloc_rank(code: str) -> int:
    if code in (" ", ""):
        return 0
    if code == "A":
        return 1
    return 2


def _pdb_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def _molecule_from_record(rec: dict, name: str) -> Molecule:
    bond_index = _pdb_conect_bonds(rec)
    return Molecule(
        np.array(rec["coords"], dtype=float), rec["elements"], name=name,
        atom_names=rec["atom_names"], resnames=rec["resnames"],
        resids=np.array(rec["resids"], dtype=int), chains=rec["chains"],
        bond_index=bond_index if len(bond_index) else None,
    )


def _parse_conect(line: str) -> list[tuple[int, int]]:
    fields = [line[6:11], line[11:16], line[16:21], line[21:26], line[26:31]]
    if not fields[0].strip():
        fields = line.split()[1:]
    if len(fields) < 2:
        return []
    try:
        source = int(fields[0])
    except ValueError:
        return []
    pairs = []
    for token in fields[1:]:
        if not token.strip():
            continue
        try:
            target = int(token)
        except ValueError:
            continue
        if target != source:
            pairs.append((source, target))
    return pairs


def _pdb_conect_bonds(rec: dict) -> np.ndarray:
    serial_to_idx = {serial: i for i, serial in enumerate(rec["serials"])}
    seen = set()
    bonds = []
    for a_serial, b_serial in rec["conect"]:
        if a_serial not in serial_to_idx or b_serial not in serial_to_idx:
            continue
        i, j = serial_to_idx[a_serial], serial_to_idx[b_serial]
        if i == j:
            continue
        pair = tuple(sorted((i, j)))
        if pair not in seen:
            seen.add(pair)
            bonds.append(pair)
    return np.array(bonds, dtype=int).reshape(-1, 2)


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


def _pdb_conect_line(a: int, b: int) -> str:
    return f"CONECT{a:5d}{b:5d}\n"


def _open(path: str, mode: str = "r"):
    """Open a file, transparently handling gzip by the ``.gz`` suffix."""
    path = os.fspath(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t")
    return open(path, mode)


def _data_extension(path: str) -> str:
    """Extension ignoring a trailing ``.gz`` (so ``a.pdb.gz`` -> ``.pdb``)."""
    path = os.fspath(path)
    base = path[:-3] if path.endswith(".gz") else path
    return os.path.splitext(base)[1].lower()


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _stem(path: str) -> str:
    path = os.fspath(path)
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = base[:-3]
    return os.path.splitext(base)[0]
