"""Fixed-size molecular descriptors for quick ML feature tables."""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .io import read

DEFAULT_ELEMENTS = (
    "H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "NA", "MG", "CA", "FE", "ZN",
)


def descriptors(
    molecule,
    *,
    elements_to_count=DEFAULT_ELEMENTS,
    distance_bins: int = 10,
    distance_range: tuple[float, float] = (0.0, 20.0),
    contact_cutoff: float = 5.0,
    residue_contact_cutoff: float = 8.0,
) -> dict:
    """Return a flat descriptor dictionary for a molecule.

    The defaults are fixed-size and suitable for small ML tables. Matrix-valued
    features such as contact maps remain available through ``mol.contact_map()``;
    this function records table-friendly summaries of them.
    """
    coords = np.asarray(molecule.coords, dtype=float)
    n_atoms = len(molecule)
    masses = molecule.masses if n_atoms else np.empty(0, dtype=float)
    desc = {
        "n_atoms": float(n_atoms),
        "n_residues": float(_n_residues(molecule)),
        "molecular_mass": float(masses.sum()) if n_atoms else 0.0,
    }

    counts = Counter(e.upper() for e in molecule.elements if e)
    for symbol in elements_to_count:
        desc[f"count_{symbol.upper()}"] = float(counts.get(symbol.upper(), 0))

    if n_atoms == 0:
        return _empty_descriptors(desc, distance_bins)

    dims = molecule.dimensions
    centroid = molecule.centroid
    center_of_mass = molecule.center_of_mass
    desc.update({
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "center_of_mass_x": float(center_of_mass[0]),
        "center_of_mass_y": float(center_of_mass[1]),
        "center_of_mass_z": float(center_of_mass[2]),
        "radius_of_gyration": molecule.radius_of_gyration,
        "dim_x": float(dims[0]),
        "dim_y": float(dims[1]),
        "dim_z": float(dims[2]),
        "bbox_volume": float(np.prod(dims)),
        "compactness": _compactness(n_atoms, dims),
    })

    inertia = inertia_tensor(molecule)
    principal_moments, principal_axes = np.linalg.eigh(inertia)
    order = np.argsort(principal_moments)
    principal_moments = principal_moments[order]
    principal_axes = principal_axes[:, order]
    desc["inertia_tensor"] = inertia.reshape(-1).astype(float).tolist()
    desc["principal_moments"] = principal_moments.astype(float).tolist()
    desc["principal_axes"] = principal_axes.reshape(-1).astype(float).tolist()
    desc["shape_anisotropy"] = shape_anisotropy(principal_moments)

    distances = _pairwise_distances(coords)
    hist, _ = np.histogram(distances, bins=distance_bins, range=distance_range)
    desc["distance_histogram"] = hist.astype(float).tolist()
    desc.update(_bond_length_summary(molecule))
    desc.update(_contact_summary(molecule, contact_cutoff))
    desc.update(_residue_contact_summary(molecule, residue_contact_cutoff))
    return desc


def inertia_tensor(molecule) -> np.ndarray:
    """Mass-weighted inertia tensor around the centre of mass."""
    coords = np.asarray(molecule.coords, dtype=float)
    if len(molecule) == 0:
        return np.zeros((3, 3), dtype=float)
    centered = coords - molecule.center_of_mass
    masses = molecule.masses
    r2 = (centered ** 2).sum(axis=1)
    tensor = np.eye(3) * np.sum(masses * r2)
    tensor -= centered.T @ (centered * masses[:, None])
    return tensor


def shape_anisotropy(principal_moments) -> float:
    """Dimensionless anisotropy from principal moments of inertia."""
    moments = np.asarray(principal_moments, dtype=float)
    denom = float(np.sum(moments ** 2))
    if denom == 0.0:
        return 0.0
    mean = float(moments.mean())
    return float(1.5 * np.sum((moments - mean) ** 2) / denom)


def featurize_many(
    paths,
    *,
    feature_names: Optional[list[str]] = None,
    return_names: bool = False,
    **descriptor_kwargs,
):
    """Read structures and return a numeric descriptor matrix.

    By default columns are the union of descriptor keys found across the input
    molecules. Pass ``feature_names`` to force a stable column order, or
    ``return_names=True`` to receive ``(X, names)``.
    """
    rows = [flatten_descriptors(descriptors(read(path), **descriptor_kwargs)) for path in paths]
    names = feature_names or sorted({key for row in rows for key in row})
    matrix = np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float)
    return (matrix, names) if return_names else matrix


def flatten_descriptors(desc: dict) -> dict[str, float]:
    """Expand list-valued descriptors into scalar columns."""
    flat = {}
    for key, value in desc.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, item in enumerate(value):
                flat[f"{key}_{i}"] = float(item)
        else:
            flat[key] = float(value)
    return flat


def _empty_descriptors(desc: dict, distance_bins: int) -> dict:
    desc.update({
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "centroid_z": 0.0,
        "center_of_mass_x": 0.0,
        "center_of_mass_y": 0.0,
        "center_of_mass_z": 0.0,
        "radius_of_gyration": 0.0,
        "dim_x": 0.0,
        "dim_y": 0.0,
        "dim_z": 0.0,
        "bbox_volume": 0.0,
        "compactness": 0.0,
        "inertia_tensor": [0.0] * 9,
        "principal_moments": [0.0] * 3,
        "principal_axes": [0.0] * 9,
        "shape_anisotropy": 0.0,
        "distance_histogram": [0.0] * distance_bins,
        "bond_count": 0.0,
        "bond_length_mean": 0.0,
        "bond_length_std": 0.0,
        "bond_length_min": 0.0,
        "bond_length_max": 0.0,
        "atom_contact_count": 0.0,
        "atom_contact_density": 0.0,
        "residue_contact_count": 0.0,
        "residue_contact_density": 0.0,
    })
    return desc


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    if n < 2:
        return np.empty(0, dtype=float)
    i, j = np.triu_indices(n, k=1)
    return np.linalg.norm(coords[i] - coords[j], axis=1)


def _bond_length_summary(molecule) -> dict[str, float]:
    bonds = molecule.bonds()
    if len(bonds) == 0:
        return {
            "bond_count": 0.0,
            "bond_length_mean": 0.0,
            "bond_length_std": 0.0,
            "bond_length_min": 0.0,
            "bond_length_max": 0.0,
        }
    lengths = np.linalg.norm(molecule.coords[bonds[:, 0]] - molecule.coords[bonds[:, 1]], axis=1)
    return {
        "bond_count": float(len(lengths)),
        "bond_length_mean": float(lengths.mean()),
        "bond_length_std": float(lengths.std()),
        "bond_length_min": float(lengths.min()),
        "bond_length_max": float(lengths.max()),
    }


def _contact_summary(molecule, cutoff: float) -> dict[str, float]:
    contacts = molecule.contacts(cutoff=cutoff)
    possible = len(molecule) * (len(molecule) - 1) / 2
    return {
        "atom_contact_count": float(len(contacts)),
        "atom_contact_density": float(len(contacts) / possible) if possible else 0.0,
    }


def _residue_contact_summary(molecule, cutoff: float) -> dict[str, float]:
    if len(molecule.resids) == 0:
        return {"residue_contact_count": 0.0, "residue_contact_density": 0.0}
    try:
        matrix = molecule.contact_map(cutoff=cutoff, level="residue").matrix
    except ValueError:
        return {"residue_contact_count": 0.0, "residue_contact_density": 0.0}
    n = len(matrix)
    possible = n * (n - 1) / 2
    count = float(np.triu(matrix.astype(bool), k=1).sum())
    return {
        "residue_contact_count": count,
        "residue_contact_density": float(count / possible) if possible else 0.0,
    }


def _n_residues(molecule) -> int:
    if len(molecule.resids) == 0:
        return 0
    return sum(1 for _ in molecule.residue_groups())


def _compactness(n_atoms: int, dims: np.ndarray) -> float:
    volume = float(np.prod(dims))
    return float(n_atoms / volume) if volume > 0.0 else 0.0
