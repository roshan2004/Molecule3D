"""3D plotting of molecules with matplotlib."""

from __future__ import annotations

import warnings

import numpy as np

from . import elements


def plot(
    molecule,
    show_bonds: bool | None = None,
    bond_tolerance: float = 1.2,
    ax=None,
    show: bool = True,
):
    """Scatter-plot atoms in 3D, coloured by element with an equal aspect ratio.

    Bonds are drawn when ``show_bonds`` is true, or, when left as ``None``,
    automatically for molecules small enough to infer bonds cheaply.

    Returns the matplotlib ``Axes3D``. Pass ``show=False`` to suppress
    ``plt.show()`` (useful for tests or saving to a file).
    """
    import matplotlib.pyplot as plt  # imported lazily so the core has no GUI dep

    coords = molecule.coords
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

    colors = [elements.color(e) for e in molecule.elements]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=20, depthshade=True)

    if show_bonds is None:
        show_bonds = len(molecule) <= 2000
    if show_bonds and len(molecule) > 1:
        try:
            for i, j in molecule.bonds(tolerance=bond_tolerance):
                seg = coords[[i, j]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.5", linewidth=1.0)
        except ValueError as exc:
            warnings.warn(f"skipping bonds: {exc}", stacklevel=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if molecule.name:
        ax.set_title(molecule.name)
    _set_equal_aspect(ax, coords)

    if show:
        plt.show()
    return ax


def _set_equal_aspect(ax, coords: np.ndarray) -> None:
    """Force equal scaling on all axes so the molecule isn't distorted."""
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    centers = (maxs + mins) / 2
    radius = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
