"""3D visualization of molecules: matplotlib, py3Dmol, and GIF export."""

from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np

from . import elements


def plot(
    molecule,
    show_bonds: Optional[bool] = None,
    bond_tolerance: float = 1.2,
    color_by: str = "element",
    scale: float = 60.0,
    ax=None,
    show: bool = True,
):
    """Scatter-plot atoms in 3D with an equal aspect ratio.

    ``color_by`` selects the colouring: ``"element"`` (CPK), ``"chain"``,
    ``"residue"`` (categorical palette), or ``"ss"`` (secondary structure, via
    a simplified DSSP). Atom sizes scale with covalent radius.
    Bonds are drawn when ``show_bonds`` is true, or, when ``None``, automatically
    for molecules small enough to infer bonds cheaply. Returns the ``Axes3D``;
    pass ``show=False`` to suppress ``plt.show()``.
    """
    import matplotlib.pyplot as plt  # imported lazily so the core has no GUI dep

    coords = molecule.coords
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

    colors = _colors(molecule, color_by)
    sizes = np.array([elements.covalent_radius(e) for e in molecule.elements]) * scale
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, depthshade=True)

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


def view(molecule, style: str = "stick", width: int = 480, height: int = 360):
    """Return an interactive py3Dmol viewer (for Jupyter notebooks).

    Requires py3Dmol (``pip install py3Dmol``). ``style`` is any py3Dmol style
    name such as ``"stick"``, ``"sphere"``, ``"line"`` or ``"cartoon"``.
    """
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - exercised only without py3Dmol
        raise ImportError(
            "view() needs py3Dmol; install it with: pip install py3Dmol"
        ) from exc
    from .io import _molecule_to_pdb_string

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(_molecule_to_pdb_string(molecule), "pdb")
    viewer.setStyle({style: {"colorscheme": "default"}})
    viewer.zoomTo()
    return viewer


def spin_gif(molecule, path: str, frames: int = 36, fps: int = 15, **plot_kwargs):
    """Render a spinning 3D view and save it as an animated GIF.

    Rotates a full turn about the vertical axis over ``frames`` steps. Requires
    Pillow (already a matplotlib dependency).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot(molecule, ax=ax, show=False, **plot_kwargs)

    def update(i):
        ax.view_init(elev=20, azim=i * 360 / frames)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_contact_map(contact_map, ax=None, cmap=None, show: bool = True):
    """Draw a :class:`~molscope.contactmap.ContactMap` as a heatmap.

    Booleans render as a binary map; ensemble frequencies render with a colour
    scale and a colourbar. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    mat = contact_map.matrix
    freq = contact_map.is_frequency
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        mat, origin="lower", interpolation="nearest", vmin=0, vmax=1,
        cmap=cmap or ("viridis" if freq else "Greys"),
    )

    unit = "residue" if contact_map.level == "residue" else "atom"
    ax.set_xlabel(f"{unit} index")
    ax.set_ylabel(f"{unit} index")
    label = "contact frequency" if freq else f"contact (< {contact_map.cutoff} Å)"
    ax.figure.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_title(f"{unit} contact map ({contact_map.cutoff} Å)")

    if show:
        plt.show()
    return ax


def plot_rmsd_heatmap(matrix, order=None, ax=None, cmap="viridis", show: bool = True):
    """Draw a pairwise-RMSD matrix as a heatmap (angstrom).

    Pass ``order`` (e.g. ``clustering.order``) to reorder rows/columns so
    clusters appear as blocks along the diagonal. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    if order is not None:
        order = np.asarray(order)
        matrix = matrix[np.ix_(order, order)]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_xlabel("model")
    ax.set_ylabel("model")
    ax.figure.colorbar(im, ax=ax, label="RMSD (Å)", fraction=0.046, pad=0.04)
    ax.set_title("pairwise RMSD")
    if show:
        plt.show()
    return ax


def _colors(molecule, color_by: str):
    if color_by == "element":
        return [elements.color(e) for e in molecule.elements]
    if color_by == "ss":
        from . import dssp

        return [dssp.SS_COLORS[c] for c in dssp.per_atom_ss(molecule)]
    if color_by == "chain":
        keys = molecule.chains
    elif color_by == "residue":
        keys = [str(r) for r in molecule.resids] if len(molecule.resids) else []
    else:
        raise ValueError(f"unknown color_by {color_by!r}")
    if not keys:
        raise ValueError(f"no {color_by} information to colour by")
    return _categorical_colors(keys)


def _categorical_colors(keys):
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab20").colors
    cycle = {}
    wheel = itertools.cycle(palette)
    return [cycle.setdefault(k, next(wheel)) for k in keys]


def _set_equal_aspect(ax, coords: np.ndarray) -> None:
    """Force equal scaling on all axes so the molecule isn't distorted."""
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    centers = (maxs + mins) / 2
    radius = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
