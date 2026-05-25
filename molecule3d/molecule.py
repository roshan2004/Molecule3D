"""The :class:`Molecule` value type and its geometric operations.

Coordinates are held as an ``(N, 3)`` numpy array. Optional per-atom metadata
(atom name, residue name, residue id, chain) travels alongside, enabling
selections such as ``mol.backbone()`` or ``mol.select(chain="A")``.

Transformations return a new ``Molecule`` rather than mutating in place, so
chains like ``mol.centered().rotate("z", 90)`` read top to bottom and never
alias state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import elements

# Above this size the dense O(n^2) bond search is refused; install scipy for the
# KD-tree path (pip install 'molecule3d[fast]') to handle larger structures.
_DENSE_BOND_LIMIT = 8000

_BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass(frozen=True, eq=False)
class Molecule:
    coords: np.ndarray
    elements: list[str] = field(default_factory=list)
    name: str = ""
    # Optional per-atom metadata; empty when the source format carries none.
    atom_names: list[str] = field(default_factory=list)
    resnames: list[str] = field(default_factory=list)
    resids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    chains: list[str] = field(default_factory=list)

    def __post_init__(self):
        coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "resids", np.asarray(self.resids, dtype=int))
        if not self.elements:
            object.__setattr__(self, "elements", [""] * len(coords))
        for name in ("elements", "atom_names", "resnames", "chains"):
            seq = getattr(self, name)
            if seq and len(seq) != len(coords):
                raise ValueError(f"{len(seq)} {name} for {len(coords)} coordinates")
        if len(self.resids) and len(self.resids) != len(coords):
            raise ValueError(f"{len(self.resids)} resids for {len(coords)} coordinates")

    def __len__(self) -> int:
        return len(self.coords)

    def __eq__(self, other) -> bool:
        # Auto-generated dataclass __eq__ can't compare the numpy field; do it
        # explicitly. coords are mutable in place, so instances stay unhashable.
        if not isinstance(other, Molecule):
            return NotImplemented
        return (
            self.name == other.name
            and self.elements == other.elements
            and np.array_equal(self.coords, other.coords)
        )

    __hash__ = None

    def __getitem__(self, selector) -> Molecule:
        """``mol[mask]`` / ``mol[indices]`` -> a subset molecule (see :meth:`take`)."""
        return self.take(selector)

    @property
    def has_topology(self) -> bool:
        """True if per-atom names/residues/chains were parsed."""
        return bool(self.atom_names)

    # -- selection ----------------------------------------------------------

    def take(self, selector) -> Molecule:
        """Return the subset given by a boolean mask or an array of indices."""
        idx = np.arange(len(self))[np.asarray(selector)]

        def sub(seq):
            return [seq[i] for i in idx] if seq else []

        return replace(
            self,
            coords=self.coords[idx],
            elements=sub(self.elements),
            atom_names=sub(self.atom_names),
            resnames=sub(self.resnames),
            resids=self.resids[idx] if len(self.resids) else self.resids,
            chains=sub(self.chains),
        )

    def select(
        self,
        element=None,
        chain=None,
        resname=None,
        atom_name=None,
        resid=None,
    ) -> Molecule:
        """Return the atoms matching every supplied criterion.

        Each of ``element``/``chain``/``resname``/``atom_name`` accepts a single
        value or a collection. ``resid`` accepts an int, a collection of ints,
        or a ``(low, high)`` inclusive range. Selecting on metadata the molecule
        lacks raises ``ValueError``.
        """
        mask = np.ones(len(self), dtype=bool)
        mask &= self._field_mask(self.elements, element, "element", upper=True)
        mask &= self._field_mask(self.chains, chain, "chain")
        mask &= self._field_mask(self.resnames, resname, "residue", upper=True)
        mask &= self._field_mask(self.atom_names, atom_name, "atom name", upper=True)
        if resid is not None:
            mask &= self._resid_mask(resid)
        return self.take(mask)

    def backbone(self) -> Molecule:
        """Protein backbone atoms (N, CA, C, O)."""
        return self.select(atom_name=_BACKBONE_ATOMS)

    def alpha_carbons(self) -> Molecule:
        """Alpha-carbon (CA) atoms, the usual basis for protein RMSD."""
        return self.select(atom_name="CA")

    def _field_mask(self, values, criterion, label, upper=False):
        if criterion is None:
            return np.ones(len(self), dtype=bool)
        if not values:
            raise ValueError(f"no {label} information in this molecule")
        wanted = {criterion} if isinstance(criterion, str) else set(criterion)
        if upper:
            wanted = {w.upper() for w in wanted}
            return np.array([v.upper() in wanted for v in values], dtype=bool)
        return np.array([v in wanted for v in values], dtype=bool)

    def _resid_mask(self, resid):
        if len(self.resids) == 0:
            raise ValueError("no residue-id information in this molecule")
        if isinstance(resid, tuple) and len(resid) == 2:
            low, high = resid
            return (self.resids >= low) & (self.resids <= high)
        wanted = [resid] if isinstance(resid, int) else list(resid)
        return np.isin(self.resids, wanted)

    # -- geometry -----------------------------------------------------------

    @property
    def masses(self) -> np.ndarray:
        """Per-atom atomic weights (g/mol)."""
        return np.array([elements.mass(e) for e in self.elements])

    @property
    def centroid(self) -> np.ndarray:
        """Geometric centre (mean of all atom positions)."""
        return self.coords.mean(axis=0)

    @property
    def center_of_mass(self) -> np.ndarray:
        """Mass-weighted centre of the molecule."""
        m = self.masses
        return (m[:, None] * self.coords).sum(axis=0) / m.sum()

    @property
    def radius_of_gyration(self) -> float:
        """Mass-weighted radius of gyration (angstrom)."""
        m = self.masses
        d2 = ((self.coords - self.center_of_mass) ** 2).sum(axis=1)
        return float(np.sqrt((m * d2).sum() / m.sum()))

    @property
    def dimensions(self) -> np.ndarray:
        """Axis-aligned bounding-box size (dx, dy, dz) in angstrom."""
        return self.coords.max(axis=0) - self.coords.min(axis=0)

    @property
    def formula(self) -> str:
        """Hill-order molecular formula, e.g. ``"C6 H12 O6"``."""
        from collections import Counter

        counts = Counter(e.capitalize() for e in self.elements if e)
        if not counts:
            return ""
        ordered = []
        for sym in ("C", "H"):
            if sym in counts:
                ordered.append((sym, counts.pop(sym)))
        ordered += sorted(counts.items())
        return " ".join(f"{s}{n}" if n > 1 else s for s, n in ordered)

    # -- transforms (each returns a new Molecule) ---------------------------

    def translate(self, vector) -> Molecule:
        """Return a copy shifted by ``vector`` (dx, dy, dz)."""
        return replace(self, coords=self.coords + np.asarray(vector, dtype=float))

    def centered(self, weighted: bool = False) -> Molecule:
        """Return a copy with its centre at the origin.

        By default the geometric centroid is used; pass ``weighted=True`` to
        centre on the mass-weighted centre of mass.
        """
        origin = self.center_of_mass if weighted else self.centroid
        return replace(self, coords=self.coords - origin)

    def rotate(self, axis, angle_deg: float) -> Molecule:
        """Return a copy rotated ``angle_deg`` degrees about ``axis``.

        ``axis`` may be ``"x"``, ``"y"``, ``"z"`` or any 3-vector. Rotation is
        about the centroid so the molecule spins in place.
        """
        vec = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }.get(axis, axis)
        rot = _rotation_matrix(np.asarray(vec, dtype=float), np.radians(angle_deg))
        center = self.centroid
        rotated = (self.coords - center) @ rot.T + center
        return replace(self, coords=rotated)

    def superpose(self, reference: Molecule) -> Molecule:
        """Return a copy optimally rotated/translated onto ``reference``.

        Uses the Kabsch algorithm (least-squares rigid-body fit). Requires the
        same number of atoms, matched by index.
        """
        if len(self) != len(reference):
            raise ValueError(f"atom count mismatch: {len(self)} vs {len(reference)}")
        p = self.coords - self.centroid
        q = reference.coords - reference.centroid
        u, _, vt = np.linalg.svd(p.T @ q)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        aligned = p @ rot.T + reference.centroid
        return replace(self, coords=aligned)

    # -- measurements & analysis -------------------------------------------

    def distance(self, i: int, j: int) -> float:
        """Distance between atoms ``i`` and ``j`` (angstrom)."""
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def angle(self, i: int, j: int, k: int) -> float:
        """Angle in degrees at atom ``j`` formed by ``i``-``j``-``k``."""
        a = self.coords[i] - self.coords[j]
        b = self.coords[k] - self.coords[j]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def dihedral(self, a: int, b: int, c: int, d: int) -> float:
        """Dihedral (torsion) angle in degrees about the ``b``-``c`` bond."""
        v0 = self.coords[a] - self.coords[b]
        v1 = self.coords[c] - self.coords[b]
        v2 = self.coords[d] - self.coords[c]
        v1 = v1 / np.linalg.norm(v1)
        v = v0 - np.dot(v0, v1) * v1
        w = v2 - np.dot(v2, v1) * v1
        x = np.dot(v, w)
        y = np.dot(np.cross(v1, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    def distance_matrix(self) -> np.ndarray:
        """Full ``(N, N)`` pairwise distance matrix (angstrom)."""
        deltas = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((deltas ** 2).sum(axis=-1))

    def contacts(self, cutoff: float = 5.0) -> np.ndarray:
        """Atom index pairs ``(i, j)`` closer than ``cutoff`` angstrom."""
        n = len(self)
        if n < 2:
            return np.empty((0, 2), dtype=int)
        try:
            from scipy.spatial import cKDTree

            return cKDTree(self.coords).query_pairs(cutoff, output_type="ndarray")
        except ImportError:
            dist = self.distance_matrix()
            i, j = np.where(np.triu(dist < cutoff, k=1))
            return np.stack([i, j], axis=1)

    def rmsd(self, other: Molecule, align: bool = False) -> float:
        """Root-mean-square deviation from ``other`` (matched by index).

        With ``align=True`` the molecules are Kabsch-superposed first, giving
        the minimum RMSD over all rigid-body orientations.
        """
        if len(self) != len(other):
            raise ValueError(f"atom count mismatch: {len(self)} vs {len(other)}")
        a = self.superpose(other).coords if align else self.coords
        return float(np.sqrt(((a - other.coords) ** 2).sum() / len(self)))

    def bonds(self, tolerance: float = 1.2) -> np.ndarray:
        """Infer bonds as index pairs ``(i, j)``.

        Two atoms bond when their separation is within ``tolerance`` times the
        sum of their covalent radii. Returns an ``(M, 2)`` int array.

        Uses ``scipy.spatial.cKDTree`` when available (scales to large
        structures); otherwise falls back to a dense search that is refused
        above ``_DENSE_BOND_LIMIT`` atoms.
        """
        n = len(self.coords)
        if n < 2:
            return np.empty((0, 2), dtype=int)
        radii = np.array([elements.covalent_radius(e) for e in self.elements])

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None

        if cKDTree is not None:
            tree = cKDTree(self.coords)
            cand = tree.query_pairs(tolerance * 2 * radii.max(), output_type="ndarray")
            if len(cand) == 0:
                return np.empty((0, 2), dtype=int)
            i, j = cand[:, 0], cand[:, 1]
        else:
            if n > _DENSE_BOND_LIMIT:
                raise ValueError(
                    f"{n} atoms exceeds the dense bond limit ({_DENSE_BOND_LIMIT}); "
                    "install scipy (pip install 'molecule3d[fast]') for large "
                    "structures."
                )
            i, j = np.triu_indices(n, k=1)

        dist = np.linalg.norm(self.coords[i] - self.coords[j], axis=1)
        cutoff = tolerance * (radii[i] + radii[j])
        keep = dist < cutoff
        return np.stack([i[keep], j[keep]], axis=1)

    def summary(self) -> str:
        """One-line human-readable description of the molecule."""
        parts = [f"{self.name or 'molecule'}: {len(self)} atoms"]
        if self.formula:
            parts.append(f"formula {self.formula}")
        if self.chains:
            parts.append(f"chains {','.join(sorted(set(self.chains)))}")
        dx, dy, dz = self.dimensions
        parts.append(f"size {dx:.1f}x{dy:.1f}x{dz:.1f} A")
        return " | ".join(parts)

    # -- graph export -------------------------------------------------------

    def to_graph(self, tolerance: float = 1.2, bonds=None):
        """Build a :class:`molecule3d.graph.MolecularGraph` from this molecule.

        Bonds are inferred from covalent radii (see :meth:`bonds`) unless an
        explicit ``(E, 2)`` array of index pairs is passed. Node and edge
        attributes (element, residue, chain, distance, ...) are carried along.
        """
        from .graph import MolecularGraph

        edges = self.bonds(tolerance) if bonds is None else np.asarray(bonds, dtype=int)
        edges = edges.reshape(-1, 2)
        if len(edges):
            dist = np.linalg.norm(self.coords[edges[:, 0]] - self.coords[edges[:, 1]], axis=1)
        else:
            dist = np.empty(0, dtype=float)
        return MolecularGraph(
            coords=self.coords, elements=self.elements, edges=edges,
            edge_distances=dist, edge_types=np.ones(len(edges)),
            atom_names=self.atom_names, resnames=self.resnames,
            resids=self.resids, chains=self.chains, name=self.name,
        )

    def to_networkx(self, **kwargs):
        """Shortcut for ``self.to_graph(...).to_networkx()``."""
        return self.to_graph(**kwargs).to_networkx()

    def to_pyg_data(self, **kwargs):
        """Shortcut for ``self.to_graph(...).to_pyg_data()`` (PyTorch Geometric)."""
        return self.to_graph(**kwargs).to_pyg_data()

    def to_dgl_graph(self, **kwargs):
        """Shortcut for ``self.to_graph(...).to_dgl_graph()`` (DGL)."""
        return self.to_graph(**kwargs).to_dgl_graph()

    def plot(self, **kwargs):
        """Render the molecule in 3D. See :func:`molecule3d.plotting.plot`."""
        from .plotting import plot

        return plot(self, **kwargs)

    def view(self, **kwargs):
        """Interactive py3Dmol viewer. See :func:`molecule3d.plotting.view`."""
        from .plotting import view

        return view(self, **kwargs)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``angle`` radians about ``axis``."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
