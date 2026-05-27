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
from typing import Any, Optional

import numpy as np

from . import elements

# Above this size the dense O(n^2) bond search is refused; install scipy for the
# KD-tree path (pip install 'molscope[fast]') to handle larger structures.

_BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass(frozen=True)
class UnitCell:
    """Crystallographic unit cell parameters.

    Lengths ``a``, ``b``, ``c`` in angstrom; angles ``alpha``, ``beta``,
    ``gamma`` in degrees.
    """

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def lattice_matrix(self) -> np.ndarray:
        """Return the (3, 3) matrix of lattice vectors as rows."""
        alpha, beta, gamma = np.radians([self.alpha, self.beta, self.gamma])
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        # Volume of a parallelepiped with unit edges
        v = np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
        )

        return np.array([
            [self.a, 0, 0],
            [self.b * cos_gamma, self.b * sin_gamma, 0],
            [self.c * cos_beta, self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, self.c * v / sin_gamma]
        ])

    def __repr__(self) -> str:
        return (
            f"UnitCell(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, "
            f"alpha={self.alpha:.2f}, beta={self.beta:.2f}, gamma={self.gamma:.2f})"
        )


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
    # Per-atom flag: True for atoms from PDB HETATM / mmCIF group_PDB=HETATM
    # records (ligands, water, ions). Empty when the source carries no record
    # type. Distinguishes polymer atoms from hetero groups for binding-site work.
    hetero: list[bool] = field(default_factory=list)
    # Optional explicit bonds as an (E, 2) index array. When set, bonds() returns
    # these instead of inferring from geometry (used by coarse-graining and file
    # formats that carry connectivity).
    bond_index: Optional[np.ndarray] = None
    bond_orders: Optional[np.ndarray] = None
    formal_charges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    unit_cell: Optional[UnitCell] = None
    _mapping_report: Optional[Any] = field(default=None, repr=False, compare=False)

    # Track selection lineage to enable boolean logic (e.g. mol1 & mol2).
    _parent: Optional[Molecule] = field(default=None, repr=False, compare=False)
    _indices: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "resids", np.asarray(self.resids, dtype=int))
        object.__setattr__(self, "formal_charges", np.asarray(self.formal_charges, dtype=int))
        if self.bond_index is not None:
            object.__setattr__(
                self, "bond_index", np.asarray(self.bond_index, dtype=int).reshape(-1, 2)
            )
        if self.bond_orders is not None:
            if self.bond_index is None:
                raise ValueError("bond_orders require bond_index")
            orders = np.asarray(self.bond_orders, dtype=float).reshape(-1)
            if len(orders) != len(self.bond_index):
                raise ValueError(
                    f"{len(orders)} bond_orders for {len(self.bond_index)} bonds"
                )
            object.__setattr__(self, "bond_orders", orders)
        if not self.elements:
            object.__setattr__(self, "elements", [""] * len(coords))
        for name in ("elements", "atom_names", "resnames", "chains", "hetero"):
            seq = getattr(self, name)
            if seq and len(seq) != len(coords):
                raise ValueError(f"{len(seq)} {name} for {len(coords)} coordinates")
        if len(self.resids) and len(self.resids) != len(coords):
            raise ValueError(f"{len(self.resids)} resids for {len(coords)} coordinates")
        if len(self.formal_charges) and len(self.formal_charges) != len(coords):
            raise ValueError(
                f"{len(self.formal_charges)} formal_charges for {len(coords)} coordinates"
            )

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
            and np.array_equal(self.formal_charges, other.formal_charges)
        )

    __hash__ = None

    def __getitem__(self, selector) -> Molecule:
        """``mol[mask]`` / ``mol[indices]`` -> a subset molecule (see :meth:`take`)."""
        return self.take(selector)

    def __and__(self, other: Molecule) -> Molecule:
        """Intersection: atoms present in both subsets of the same parent."""
        self._check_same_parent(other)
        mask = np.isin(self._indices, other._indices)
        return self.take(mask)

    def __or__(self, other: Molecule) -> Molecule:
        """Union: atoms present in either subset of the same parent."""
        self._check_same_parent(other)
        # We need to maintain the parent's order.
        combined_idx = np.union1d(self._indices, other._indices)
        return self._parent.take(combined_idx)

    def __sub__(self, other: Molecule) -> Molecule:
        """Difference: atoms in this subset but not the other."""
        self._check_same_parent(other)
        mask = ~np.isin(self._indices, other._indices)
        return self.take(mask)

    def __invert__(self) -> Molecule:
        """Complement: atoms in the parent molecule NOT in this subset."""
        if self._parent is None:
            # Complement of a root molecule is an empty molecule.
            return self.take(np.zeros(len(self), dtype=bool))
        mask = np.ones(len(self._parent), dtype=bool)
        mask[self._indices] = False
        return self._parent.take(mask)

    def _check_same_parent(self, other: Molecule):
        if not isinstance(other, Molecule):
            raise TypeError(f"cannot combine Molecule with {type(other).__name__}")
        if self._parent is None or self._parent is not other._parent:
            raise ValueError("boolean operations only supported on subsets of the same molecule")

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

        bond_index, bond_orders = self._subset_bonds(idx)
        return replace(
            self,
            coords=self.coords[idx],
            elements=sub(self.elements),
            atom_names=sub(self.atom_names),
            resnames=sub(self.resnames),
            resids=self.resids[idx] if len(self.resids) else self.resids,
            chains=sub(self.chains),
            hetero=sub(self.hetero),
            formal_charges=(
                self.formal_charges[idx] if len(self.formal_charges) else self.formal_charges
            ),
            bond_index=bond_index,
            bond_orders=bond_orders,
            _mapping_report=None,
            _parent=self,
            _indices=idx,
        )


    def _subset_bonds(self, idx):
        """Restrict explicit bonds to a kept-atom index set and renumber them."""
        if self.bond_index is None:
            return None, None
        remap = {old: new for new, old in enumerate(idx)}
        kept, orders = [], []
        source_orders = (
            self.bond_orders if self.bond_orders is not None
            else np.ones(len(self.bond_index), dtype=float)
        )
        for (i, j), order in zip(self.bond_index, source_orders):
            if i in remap and j in remap:
                kept.append((remap[i], remap[j]))
                orders.append(float(order))
        bond_index = np.array(kept, dtype=int).reshape(-1, 2)
        bond_orders = np.array(orders, dtype=float) if self.bond_orders is not None else None
        return bond_index, bond_orders

    def select(
        self,
        element=None,
        chain=None,
        resname=None,
        atom_name=None,
        resid=None,
        hetero=None,
    ) -> Molecule:
        """Return the atoms matching every supplied criterion.

        Each of ``element``/``chain``/``resname``/``atom_name`` accepts a single
        value or a collection. ``resid`` accepts an int, a collection of ints,
        or a ``(low, high)`` inclusive range. ``hetero`` accepts a bool to keep
        only HETATM (``True``) or only ATOM (``False``) atoms. Selecting on
        metadata the molecule lacks raises ``ValueError``.
        """
        mask = np.ones(len(self), dtype=bool)
        mask &= self._field_mask(self.elements, element, "element", upper=True)
        mask &= self._field_mask(self.chains, chain, "chain")
        mask &= self._field_mask(self.resnames, resname, "residue", upper=True)
        mask &= self._field_mask(self.atom_names, atom_name, "atom name", upper=True)
        if resid is not None:
            mask &= self._resid_mask(resid)
        if hetero is not None:
            if not self.hetero:
                raise ValueError("no ATOM/HETATM record information in this molecule")
            mask &= np.array(self.hetero, dtype=bool) == bool(hetero)
        return self.take(mask)

    def select_within(self, radius: float, target: Any) -> Molecule:
        """Return the atoms within ``radius`` of a target.

        The target can be another ``Molecule``, a single ``(3,)`` coordinate, or
        an ``(M, 3)`` array of coordinates.
        """
        from .distance import cdist

        if isinstance(target, Molecule):
            other_coords = target.coords
        else:
            other_coords = np.asarray(target).reshape(-1, 3)

        # dists is (N, M)
        dists = cdist(self.coords, other_coords)
        mask = np.any(dists < radius, axis=1)
        return self.take(mask)


    def protein(self) -> Molecule:
        """Polymer atoms (those from ATOM records, i.e. not HETATM)."""
        return self.select(hetero=False)

    def hetero_atoms(self) -> Molecule:
        """Hetero atoms (those from HETATM records: ligands, water, ions)."""
        return self.select(hetero=True)

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

    def residue_groups(self):
        """Yield ``(atom_indices, resname, resid, chain)`` per residue, in order.

        Residues are runs of atoms sharing ``(chain, resid)``. Yields nothing if
        the molecule has no residue information.
        """
        n = len(self)
        if len(self.resids) == 0 or n == 0:
            return
        chains = self.chains or [""] * n
        resnames = self.resnames or [""] * n
        resids = self.resids
        start = 0
        for i in range(1, n + 1):
            if i == n or chains[i] != chains[i - 1] or resids[i] != resids[i - 1]:
                yield (
                    list(range(start, i)), resnames[start],
                    int(resids[start]), chains[start],
                )
                start = i

    # -- geometry -----------------------------------------------------------

    @property
    def masses(self) -> np.ndarray:
        """Per-atom atomic weights (g/mol)."""
        return np.array([elements.mass(e) for e in self.elements])

    @property
    def centroid(self) -> np.ndarray:
        """Geometric centre: the unweighted mean of atom positions.

        Contrast :attr:`center_of_mass`, which weights by atomic mass; the two
        differ when heavy atoms sit off-centre.
        """
        return self.coords.mean(axis=0)

    @property
    def center_of_mass(self) -> np.ndarray:
        """Mass-weighted centre: ``sum(m_i r_i) / sum(m_i)``."""
        m = self.masses
        return (m[:, None] * self.coords).sum(axis=0) / m.sum()

    @property
    def radius_of_gyration(self) -> float:
        """Mass-weighted RMS distance of atoms from the centre of mass (angstrom).

        ``Rg = sqrt(sum(m_i |r_i - R_com|^2) / sum(m_i))`` — a compactness
        measure: smaller for globular structures, larger for extended ones.
        """
        m = self.masses
        d2 = ((self.coords - self.center_of_mass) ** 2).sum(axis=1)
        return float(np.sqrt((m * d2).sum() / m.sum()))

    @property
    def dimensions(self) -> np.ndarray:
        """Axis-aligned bounding-box size (dx, dy, dz) in angstrom."""
        return self.coords.max(axis=0) - self.coords.min(axis=0)

    def inertia_tensor(self) -> np.ndarray:
        """Mass-weighted moment-of-inertia tensor ``(3, 3)`` about the centre of mass.

        ``I = sum_i m_i (|r_i|^2 I_3 - r_i r_i^T)`` with ``r_i`` relative to the
        centre of mass. Its eigenvectors are the principal axes and eigenvalues
        the principal moments (see :meth:`principal_moments`).
        """
        from .descriptors import inertia_tensor

        return inertia_tensor(self)

    def principal_moments(self) -> np.ndarray:
        """Principal moments of inertia ``(3,)``, ascending.

        The eigenvalues of :meth:`inertia_tensor`; they describe the mass
        distribution along the principal axes (equal for a sphere, two-large-
        one-small for a rod, etc.).
        """
        return np.linalg.eigvalsh(self.inertia_tensor())

    def principal_axes(self) -> np.ndarray:
        """Principal axes as columns of a ``(3, 3)`` matrix, ordered by ascending moment.

        The eigenvectors of :meth:`inertia_tensor`; column ``k`` is the axis
        whose moment is ``principal_moments()[k]``.
        """
        moments, axes = np.linalg.eigh(self.inertia_tensor())
        return axes[:, np.argsort(moments)]

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
        p = self.coords - self.centroid            # centre both sets at the origin
        q = reference.coords - reference.centroid
        u, _, vt = np.linalg.svd(p.T @ q)          # SVD of the cross-covariance
        d = np.sign(np.linalg.det(vt.T @ u.T))     # correct for a reflection
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T  # optimal rotation
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

    def distance_matrix(
        self,
        backend: str = "numpy",
        device: str | None = None,
        as_numpy: bool = True,
    ):
        """Full ``(N, N)`` pairwise distance matrix (angstrom).

        ``backend`` may be ``"numpy"`` (default), ``"torch"``, ``"cupy"`` or
        ``"auto"``. Torch and CuPy are optional; use them when you already have
        a CPU/GPU array stack installed. Results are converted to NumPy by
        default for compatibility; pass ``as_numpy=False`` to keep a backend
        array.
        """
        from .distance import distance_matrix

        return distance_matrix(
            self.coords, backend=backend, device=device, as_numpy=as_numpy,
            unit_cell=self.unit_cell,
        )

    def contacts(
        self,
        cutoff: float = 5.0,
        backend: str = "scipy",
        device: str | None = None,
    ) -> np.ndarray:
        """Atom index pairs ``(i, j)`` closer than ``cutoff`` angstrom.

        ``backend="scipy"`` uses a KD-tree when SciPy is installed. Without SciPy the
        fallback uses an efficient O(N) cell-list algorithm, so it avoids
        materializing the full distance matrix. Dense backends (``"numpy"``,
        ``"torch"``, ``"cupy"``, ``"auto"``) materialize a full contact matrix first,
        which is convenient for CPU/GPU dense workflows but scales as ``O(N^2)``
        memory.
        """
        n = len(self)
        if n < 2:
            return np.empty((0, 2), dtype=int)
        if cutoff <= 0.0:
            return np.empty((0, 2), dtype=int)
        if backend in {"numpy", "torch", "cupy", "auto"}:
            from .distance import contact_matrix, contacts_from_matrix

            mat = contact_matrix(
                self.coords, cutoff=cutoff, backend=backend, device=device,
                unit_cell=self.unit_cell,
            )
            return contacts_from_matrix(mat)
        if backend != "scipy":
            raise ValueError(
                "backend must be 'scipy', 'numpy', 'torch', 'cupy' or 'auto', "
                f"got {backend!r}"
            )
        try:
            # KD-tree doesn't natively support general PBC MIC
            if self.unit_cell is not None:
                raise ImportError("mocking for PBC")
            from scipy.spatial import cKDTree

            return cKDTree(self.coords).query_pairs(cutoff, output_type="ndarray")
        except ImportError:
            from .distance import find_contacts

            return find_contacts(self.coords, cutoff, unit_cell=self.unit_cell)


    def contact_count(
        self,
        cutoff: float = 5.0,
        backend: str = "scipy",
        device: str | None = None,
    ) -> int:
        """Count atom pairs closer than ``cutoff`` without returning the pairs."""
        n = len(self)
        if n < 2 or cutoff <= 0.0:
            return 0
        if backend in {"numpy", "torch", "cupy", "auto"}:
            return int(len(self.contacts(cutoff=cutoff, backend=backend, device=device)))
        if backend != "scipy":
            raise ValueError(
                "backend must be 'scipy', 'numpy', 'torch', 'cupy' or 'auto', "
                f"got {backend!r}"
            )
        try:
            if self.unit_cell is not None:
                raise ImportError("mocking for PBC")
            from scipy.spatial import cKDTree

            tree = cKDTree(self.coords)
            # count_neighbors(tree, tree) includes self-pairs and both pair
            # directions; remove the diagonal and collapse i->j / j->i.
            return max(0, int((tree.count_neighbors(tree, cutoff) - n) // 2))
        except ImportError:
            from .distance import find_contact_count

            return find_contact_count(self.coords, cutoff, unit_cell=self.unit_cell)



    def contact_map(
        self,
        cutoff: float = 8.0,
        level: str = "residue",
        method: str = "ca",
        backend: str = "numpy",
        device: str | None = None,
        min_seq_sep: int = 0,
        chain_mode: str = "all",
    ):
        """Build a contact map. See :func:`molscope.contactmap.contact_map`.

        ``level`` is ``"atom"`` or ``"residue"``; for residue level ``method`` is
        ``"ca"`` (CA-CA distance), ``"com"`` (centre of mass) or ``"min"``
        (closest inter-residue atom). ``backend`` may be ``"numpy"``,
        ``"torch"``, ``"cupy"`` or ``"auto"`` for dense distance work.
        ``min_seq_sep`` drops same-chain contacts closer than that many positions;
        ``chain_mode`` keeps ``"all"``/``"intra"``/``"inter"``-chain pairs.
        Returns a :class:`ContactMap`.
        """
        from .contactmap import contact_map

        return contact_map(
            self, cutoff=cutoff, level=level, method=method,
            backend=backend, device=device,
            min_seq_sep=min_seq_sep, chain_mode=chain_mode,
        )

    def plot_contact_map(self, cutoff: float = 8.0, level: str = "residue",
                         method: str = "ca", backend: str = "numpy",
                         device: str | None = None, **kwargs):
        """Shortcut for ``self.contact_map(...).plot()``."""
        return self.contact_map(cutoff, level, method, backend, device).plot(**kwargs)

    def plot_distance_matrix(self, backend: str = "numpy", device: str | None = None,
                             **kwargs):
        """Plot the dense pairwise distance matrix as a heatmap."""
        from .plotting import plot_distance_matrix

        return plot_distance_matrix(
            self.distance_matrix(backend=backend, device=device), **kwargs
        )

    def secondary_structure(self):
        """Assign protein secondary structure with a simplified DSSP.

        Returns a :class:`molscope.dssp.SecondaryStructure` with one code per
        backbone residue (``H``/``G``/``I`` helices, ``E``/``B`` strands, ``T``
        turn, ``S`` bend, ``-`` coil). Needs N/CA/C/O backbone atoms and residue
        metadata, so it works on proteins read from PDB/mmCIF. Colour a 3D plot
        by the assignment with ``mol.plot(color_by="ss")``.
        """
        from .dssp import assign

        return assign(self)

    def backbone_torsions(self):
        """Backbone phi/psi/omega dihedrals per residue (Ramachandran angles).

        Returns a :class:`molscope.dssp.BackboneTorsions` with ``(R,)`` arrays in
        degrees, ``NaN`` at chain termini and breaks. Needs N/CA/C/O backbone
        atoms and residue metadata (proteins read from PDB/mmCIF).
        """
        from .dssp import backbone_torsions

        return backbone_torsions(self)

    def chain_ids(self) -> list[str]:
        """Unique chain ids in first-seen order."""
        return list(dict.fromkeys(self.chains))

    def interface(self, chain_a: str, chain_b: str, cutoff: float = 5.0):
        """Residues across the ``chain_a``/``chain_b`` interface.

        Returns a :class:`molscope.contacts.Interface`. See
        :func:`molscope.contacts.interface_residues`.
        """
        from .contacts import interface_residues

        return interface_residues(self, chain_a, chain_b, cutoff=cutoff)

    def chain_contacts(self, cutoff: float = 5.0):
        """Inter-chain atom-contact counts as a
        :class:`molscope.contacts.ChainContactMatrix`."""
        from .contacts import chain_contact_matrix

        return chain_contact_matrix(self, cutoff=cutoff)

    def ligands(self, exclude_water: bool = True, exclude_ions: bool = True):
        """HETATM groups that look like ligands (skips solvent/ions by default).

        Returns a list of :class:`molscope.contacts.LigandResidue`.
        """
        from .contacts import ligands

        return ligands(self, exclude_water=exclude_water, exclude_ions=exclude_ions)

    def binding_site(self, ligand=None, cutoff: float = 4.5):
        """Protein residues surrounding a ligand HETATM group.

        Returns a :class:`molscope.contacts.BindingSite`. See
        :func:`molscope.contacts.binding_site`.
        """
        from .contacts import binding_site

        return binding_site(self, ligand=ligand, cutoff=cutoff)

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
        above ``_DENSE_BOND_LIMIT`` atoms. If the molecule carries explicit
        bonds (e.g. a coarse-grained model), those are returned directly.
        """
        if self.bond_index is not None:
            return self.bond_index
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
            from .distance import find_contacts

            # Use the same max-radius trick as the KD-tree path to find candidates,
            # then filter by the sum of radii.
            cand = find_contacts(self.coords, tolerance * 2 * radii.max())
            if len(cand) == 0:
                return np.empty((0, 2), dtype=int)
            i, j = cand[:, 0], cand[:, 1]

        dist = np.linalg.norm(self.coords[i] - self.coords[j], axis=1)

        cutoff = tolerance * (radii[i] + radii[j])
        keep = dist < cutoff
        return np.stack([i[keep], j[keep]], axis=1)

    def bond_order_array(self, tolerance: float = 1.2) -> np.ndarray:
        """Bond-order values aligned with :meth:`bonds`.

        Explicit file/topology bonds preserve their source order values where
        available. Geometrically inferred bonds have unknown order and are
        reported as ``1.0``.
        """
        if self.bond_index is not None:
            if self.bond_orders is not None:
                return self.bond_orders
            return np.ones(len(self.bond_index), dtype=float)
        return np.ones(len(self.bonds(tolerance)), dtype=float)

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

    # -- coarse-graining ----------------------------------------------------

    def coarse_grain(self, mapping="residue_com", weighted: bool = True,
                     bonds=None, return_report: bool = False):
        """Map this structure onto CG beads. See :mod:`molscope.coarsegrain`.

        ``mapping`` is ``"residue_com"``, ``"residue_centroid"``, ``"martini"``,
        a ``{resname: {bead: [atom_names]}}`` dict (by residue), or a
        ``{bead: [atom_indices]}`` dict (by index, works on any structure).
        ``bonds`` optionally defines the bead network as pairs of bead indices,
        or bead names when those names are unique. Repeated residue bead names
        such as ``BB``/``SC`` are ambiguous; use indices for those. Returns a
        new ``Molecule`` of beads with CG bonds attached, or ``(molecule,
        report)`` when ``return_report=True``.
        """
        from .coarsegrain import coarse_grain

        return coarse_grain(
            self, mapping=mapping, weighted=weighted, bonds=bonds,
            return_report=return_report,
        )

    @property
    def coarse_grain_report(self):
        """The structured :class:`~molscope.coarsegrain.CoarseGrainReport`.

        Available on molecules produced by :meth:`coarse_grain`; raises
        otherwise. Carries the per-bead atom assignment that drives
        :meth:`plot_mapping` and the mapping export helpers.
        """
        if self._mapping_report is None:
            raise ValueError("no coarse-graining report is available for this molecule")
        return self._mapping_report

    def mapping_report(self) -> str:
        """Explain how this coarse-grained molecule was mapped (text)."""
        return self.coarse_grain_report.format()

    def plot_mapping(self, atomistic: Molecule, **kwargs):
        """Visualise how this CG model maps onto its ``atomistic`` source.

        See :func:`molscope.plotting.plot_mapping`. ``atomistic`` must be the
        molecule this CG model was built from (matching atom order).
        """
        from .plotting import plot_mapping

        return plot_mapping(atomistic, self, **kwargs)

    def write_mapping(self, path: str) -> str:
        """Write this CG model's mapping to JSON. See :func:`molscope.write_cg_mapping`."""
        from .coarsegrain import write_mapping

        return write_mapping(self, path)

    def write_index(self, path: str, per_line: int = 15) -> str:
        """Write this CG model's bead assignment as a GROMACS-style ``.ndx`` file."""
        from .coarsegrain import write_index

        return write_index(self, path, per_line=per_line)

    # -- ML descriptors -----------------------------------------------------

    def descriptors(self, **kwargs) -> dict:
        """Return fixed-size molecular descriptors for quick ML features."""
        from .descriptors import descriptors

        return descriptors(self, **kwargs)

    def chemical_features(self, **kwargs):
        """Return optional RDKit-backed aromaticity, valence and charge features.

        Requires the ``chem`` extra (``pip install "molscope[chem]"``). Explicit
        SDF bond orders and formal charges are used when present; coordinate-only
        structures fall back to geometrically inferred single bonds.
        """
        from .chem import chemical_features

        return chemical_features(self, **kwargs)

    def rdkit_descriptors(self, **kwargs) -> dict[str, float]:
        """Return optional RDKit scalar molecular descriptors.

        Requires the ``chem`` extra (``pip install "molscope[chem]"``). Use
        ``descriptors(include_rdkit=True)`` to merge these into the MolScope
        native descriptor dictionary.
        """
        from .chem import rdkit_descriptors

        return rdkit_descriptors(self, **kwargs)

    # -- graph export -------------------------------------------------------

    def to_graph(
        self,
        tolerance: float = 1.2,
        bonds=None,
        bond_orders=None,
        include_chemical_features: bool = False,
    ):
        """Build a :class:`molscope.graph.MolecularGraph` from this molecule.

        Bonds are inferred from covalent radii (see :meth:`bonds`) unless an
        explicit ``(E, 2)`` array of index pairs is passed. Explicit bond orders
        from input files are preserved when available; inferred or user-supplied
        bonds default to order ``1.0`` unless ``bond_orders=`` is passed. Node
        and edge attributes (element, residue, chain, distance, ...) are carried
        along. With ``include_chemical_features=True``, optional RDKit-backed
        aromatic atom/bond flags are attached when the ``chem`` extra is
        installed.
        """
        from .graph import MolecularGraph

        aromatic_atoms = np.empty(0, dtype=bool)
        aromatic_bonds = np.empty(0, dtype=bool)
        if bonds is None:
            edges = self.bonds(tolerance)
            if self.bond_index is not None and self.bond_orders is not None:
                edge_types = self.bond_orders
            else:
                edge_types = np.ones(len(edges), dtype=float)
            if include_chemical_features:
                features = self.chemical_features()
                aromatic_atoms = features.aromatic_atoms
                aromatic_bonds = _align_bond_flags(
                    edges, features.bond_index, features.aromatic_bonds
                )
        else:
            edges = np.asarray(bonds, dtype=int)
            if bond_orders is None:
                edge_types = np.ones(len(edges), dtype=float)
            else:
                edge_types = np.asarray(bond_orders, dtype=float).reshape(-1)
        edges = edges.reshape(-1, 2)
        if len(edge_types) != len(edges):
            raise ValueError(f"{len(edge_types)} bond_orders for {len(edges)} bonds")
        if len(edges):
            dist = np.linalg.norm(self.coords[edges[:, 0]] - self.coords[edges[:, 1]], axis=1)
        else:
            dist = np.empty(0, dtype=float)
        return MolecularGraph(
            coords=self.coords, elements=self.elements, edges=edges,
            edge_distances=dist, edge_types=edge_types,
            atom_names=self.atom_names, resnames=self.resnames,
            resids=self.resids, chains=self.chains,
            formal_charges=self.formal_charges, aromatic_atoms=aromatic_atoms,
            aromatic_bonds=aromatic_bonds, name=self.name,
        )

    def to_networkx(self, **kwargs):
        """Shortcut for ``self.to_graph(...).to_networkx()``."""
        return self.to_graph(**kwargs).to_networkx()

    def to_pyg_data(
        self,
        node_preset: str = "default",
        edge_preset: str = "default",
        **kwargs,
    ):
        """Shortcut for ``self.to_graph(...).to_pyg_data()`` (PyTorch Geometric)."""
        return self.to_graph(**kwargs).to_pyg_data(
            node_preset=node_preset,
            edge_preset=edge_preset,
        )

    def to_dgl_graph(
        self,
        node_preset: str = "default",
        edge_preset: str = "default",
        **kwargs,
    ):
        """Shortcut for ``self.to_graph(...).to_dgl_graph()`` (DGL)."""
        return self.to_graph(**kwargs).to_dgl_graph(
            node_preset=node_preset,
            edge_preset=edge_preset,
        )

    def to_residue_contact_graph(
        self,
        cutoff: float = 8.0,
        method: str = "ca",
        backend: str = "numpy",
        device: str | None = None,
        min_seq_sep: int = 0,
        chain_mode: str = "all",
    ):
        """Build a residue-level spatial contact graph for graph ML.

        Residues become nodes and residue-residue contacts become edges. See
        :func:`molscope.graph.residue_contact_graph` for the construction
        methods and exporters.
        """
        from .graph import residue_contact_graph

        return residue_contact_graph(
            self,
            cutoff=cutoff,
            method=method,
            backend=backend,
            device=device,
            min_seq_sep=min_seq_sep,
            chain_mode=chain_mode,
        )

    def plot(self, **kwargs):
        """Render the molecule in 3D. See :func:`molscope.plotting.plot`."""
        from .plotting import plot

        return plot(self, **kwargs)

    def view(self, **kwargs):
        """Interactive py3Dmol viewer. See :func:`molscope.plotting.view`."""
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


def _align_bond_flags(edges: np.ndarray, source_edges: np.ndarray, flags: np.ndarray) -> np.ndarray:

    if len(edges) == 0:
        return np.empty(0, dtype=bool)
    flag_by_pair = {
        tuple(sorted((int(i), int(j)))): bool(flag)
        for (i, j), flag in zip(source_edges, flags)
    }
    return np.array([
        flag_by_pair.get(tuple(sorted((int(i), int(j)))), False)
        for i, j in edges
    ], dtype=bool)
