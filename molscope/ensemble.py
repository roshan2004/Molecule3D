"""Analysis across a set of structures, e.g. the models of an NMR ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .molecule import Molecule


def align_all(models: list[Molecule], reference: Optional[Molecule] = None) -> list[Molecule]:
    """Kabsch-superpose every model onto ``reference`` (default: the first model)."""
    ref = reference if reference is not None else models[0]
    return [m.superpose(ref) for m in models]


def average(models: list[Molecule], align: bool = True) -> Molecule:
    """Average structure over the ensemble (atoms matched by index)."""
    _check_consistent(models)
    aligned = align_all(models) if align else models
    coords = np.mean([m.coords for m in aligned], axis=0)
    return replace(models[0], coords=coords, name=f"{models[0].name} (average)")


def rmsf(models: list[Molecule], align: bool = True) -> np.ndarray:
    """Per-atom root-mean-square fluctuation about the mean position."""
    _check_consistent(models)
    aligned = align_all(models) if align else models
    stack = np.array([m.coords for m in aligned])      # (n_models, n_atoms, 3)
    mean = stack.mean(axis=0)
    return np.sqrt(((stack - mean) ** 2).sum(axis=2).mean(axis=0))


def contact_frequency(models: list[Molecule], cutoff: float = 8.0,
                      level: str = "residue", method: str = "ca"):
    """Fraction of models in which each pair is in contact (an ensemble map).

    Returns a :class:`~molscope.contactmap.ContactMap` whose matrix holds
    values in ``[0, 1]`` — the contact probability for each residue (or atom)
    pair across the ensemble. Useful for NMR variability and folding analysis.
    """
    from .contactmap import ContactMap, contact_map

    _check_consistent(models)
    maps = [contact_map(m, cutoff=cutoff, level=level, method=method) for m in models]
    freq = np.mean([cm.matrix for cm in maps], axis=0)
    first = maps[0]
    return ContactMap(freq, level=level, cutoff=cutoff,
                      labels=first.labels, resids=first.resids)


def rmsd_matrix(models: list[Molecule], align: bool = True) -> np.ndarray:
    """Symmetric ``(M, M)`` matrix of pairwise RMSDs between models."""
    _check_consistent(models)
    n = len(models)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = models[i].rmsd(models[j], align=align)
    return mat


def _check_consistent(models: list[Molecule]) -> None:
    if not models:
        raise ValueError("no models given")
    sizes = {len(m) for m in models}
    if len(sizes) != 1:
        raise ValueError(f"models have differing atom counts: {sorted(sizes)}")


@dataclass
class Clustering:
    """Result of clustering structures by RMSD.

    ``labels`` gives the 1-based cluster id of each model (same order as the
    input). ``matrix`` is the RMSD matrix used and ``linkage`` the scipy linkage.
    """

    labels: np.ndarray
    matrix: np.ndarray
    linkage: Optional[np.ndarray] = field(default=None)

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels)))

    @property
    def order(self) -> np.ndarray:
        """Model indices sorted by cluster (for a block-diagonal heatmap)."""
        return np.argsort(self.labels, kind="stable")

    def groups(self) -> dict[int, list[int]]:
        """Map each cluster id to the list of model indices it contains."""
        return {int(c): np.where(self.labels == c)[0].tolist()
                for c in np.unique(self.labels)}

    def medoid(self, cluster_id: int) -> int:
        """Index of the most central model of a cluster (min total RMSD)."""
        members = np.where(self.labels == cluster_id)[0]
        sub = self.matrix[np.ix_(members, members)]
        return int(members[sub.sum(axis=1).argmin()])

    def representatives(self) -> dict[int, int]:
        """Map each cluster id to its medoid model index."""
        return {int(c): self.medoid(int(c)) for c in np.unique(self.labels)}


def cluster(models, method: str = "hierarchical", cutoff: Optional[float] = None,
            n_clusters: Optional[int] = None, linkage: str = "average",
            align: bool = True, matrix=None) -> Clustering:
    """Cluster structures by pairwise RMSD.

    Pass ``n_clusters`` to cut the tree into a fixed number of clusters, or
    ``cutoff`` (an RMSD threshold in angstrom). With neither, a data-driven
    cutoff (the mean pairwise RMSD) is used. Reuses ``matrix`` if given, else
    computes :func:`rmsd_matrix`. Requires scipy.
    """
    if method != "hierarchical":
        raise ValueError(f"unknown method {method!r}; only 'hierarchical' is supported")

    dm = np.asarray(matrix, dtype=float) if matrix is not None else rmsd_matrix(models, align=align)
    if len(dm) < 2:
        return Clustering(labels=np.ones(len(dm), dtype=int), matrix=dm)

    try:
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage as _linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "clustering needs scipy; install it with: pip install 'molscope[fast]'"
        ) from exc

    z = _linkage(squareform(dm, checks=False), method=linkage)
    if n_clusters is not None:
        labels = fcluster(z, t=n_clusters, criterion="maxclust")
    else:
        if cutoff is None:
            cutoff = float(dm[np.triu_indices_from(dm, k=1)].mean())
        labels = fcluster(z, t=cutoff, criterion="distance")
    return Clustering(labels=labels, matrix=dm, linkage=z)
