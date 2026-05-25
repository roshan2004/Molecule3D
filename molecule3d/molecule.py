"""The :class:`Molecule` value type and its geometric operations.

Coordinates are held as an ``(N, 3)`` numpy array. Transformations return a new
``Molecule`` rather than mutating in place, so chains like
``mol.centered().rotate("z", 90)`` read top to bottom and never alias state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import elements

# Above this size the dense O(n^2) bond search is refused; install scipy for the
# KD-tree path (pip install 'molecule3d[fast]') to handle larger structures.
_DENSE_BOND_LIMIT = 8000


@dataclass(frozen=True, eq=False)
class Molecule:
    coords: np.ndarray
    elements: list[str] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        object.__setattr__(self, "coords", coords)
        if not self.elements:
            object.__setattr__(self, "elements", [""] * len(coords))
        elif len(self.elements) != len(coords):
            raise ValueError(
                f"{len(self.elements)} elements for {len(coords)} coordinates"
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
        )

    __hash__ = None

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
            raise ValueError(
                f"atom count mismatch: {len(self)} vs {len(reference)}"
            )
        p = self.coords - self.centroid
        q = reference.coords - reference.centroid
        u, _, vt = np.linalg.svd(p.T @ q)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        aligned = p @ rot.T + reference.centroid
        return replace(self, coords=aligned)

    # -- analysis -----------------------------------------------------------

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
            cand = tree.query_pairs(
                tolerance * 2 * radii.max(), output_type="ndarray"
            )
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
            iu, ju = np.triu_indices(n, k=1)
            i, j = iu, ju

        dist = np.linalg.norm(self.coords[i] - self.coords[j], axis=1)
        cutoff = tolerance * (radii[i] + radii[j])
        keep = dist < cutoff
        return np.stack([i[keep], j[keep]], axis=1)

    def plot(self, **kwargs):
        """Render the molecule in 3D. See :func:`molecule3d.plotting.plot`."""
        from .plotting import plot

        return plot(self, **kwargs)


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
