"""The :class:`Molecule` value type and its geometric operations.

Coordinates are held as an ``(N, 3)`` numpy array. Transformations return a new
``Molecule`` rather than mutating in place, so chains like
``mol.centered().rotate("z", 90)`` read top to bottom and never alias state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import elements


@dataclass(frozen=True)
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

    @property
    def centroid(self) -> np.ndarray:
        """Geometric centre (mean of all atom positions)."""
        return self.coords.mean(axis=0)

    def translate(self, vector) -> "Molecule":
        """Return a copy shifted by ``vector`` (dx, dy, dz)."""
        return replace(self, coords=self.coords + np.asarray(vector, dtype=float))

    def centered(self) -> "Molecule":
        """Return a copy translated so the centroid sits at the origin."""
        return replace(self, coords=self.coords - self.centroid)

    def rotate(self, axis, angle_deg: float) -> "Molecule":
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

    def bonds(self, tolerance: float = 1.2, max_atoms: int = 6000) -> np.ndarray:
        """Infer bonds as index pairs ``(i, j)``.

        Two atoms bond when their separation is within ``tolerance`` times the
        sum of their covalent radii. Returns an ``(M, 2)`` int array. Raises if
        the molecule is larger than ``max_atoms`` (the pairwise distance matrix
        would be too big); pass a higher limit to override.
        """
        n = len(self.coords)
        if n > max_atoms:
            raise ValueError(
                f"{n} atoms exceeds max_atoms={max_atoms}; bond inference is "
                "O(n^2). Raise max_atoms to force it."
            )
        radii = np.array([elements.covalent_radius(e) for e in self.elements])
        deltas = self.coords[:, None, :] - self.coords[None, :, :]
        dist = np.sqrt((deltas ** 2).sum(axis=-1))
        cutoff = tolerance * (radii[:, None] + radii[None, :])
        i, j = np.where((dist < cutoff) & (dist > 1e-3))
        upper = i < j
        return np.stack([i[upper], j[upper]], axis=1)

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
