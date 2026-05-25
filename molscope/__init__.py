"""molscope: read, analyse, and plot molecular structures in 3D."""

from . import coarsegrain, ensemble
from .coarsegrain import BeadMapping, BondMapping, CoarseGrainReport, DroppedAtom
from .contactmap import ContactMap
from .descriptors import descriptors, featurize_many
from .ensemble import Clustering, cluster, rmsd_matrix
from .ensemble import contact_frequency as ensemble_contact_frequency
from .graph import MolecularGraph
from .io import (
    fetch,
    read,
    read_cif,
    read_pdb,
    read_pdb_models,
    read_sdf,
    read_xyz,
    read_xyz_frames,
    write_pdb,
    write_xyz,
)
from .molecule import Molecule
from .plotting import plot_rmsd_heatmap

__all__ = [
    "Clustering",
    "BeadMapping",
    "BondMapping",
    "CoarseGrainReport",
    "ContactMap",
    "DroppedAtom",
    "Molecule",
    "MolecularGraph",
    "cluster",
    "coarsegrain",
    "descriptors",
    "ensemble",
    "ensemble_contact_frequency",
    "featurize_many",
    "fetch",
    "plot_rmsd_heatmap",
    "read",
    "read_cif",
    "read_pdb",
    "read_pdb_models",
    "read_sdf",
    "read_xyz",
    "read_xyz_frames",
    "rmsd_matrix",
    "write_pdb",
    "write_xyz",
]
__version__ = "0.6.0"
