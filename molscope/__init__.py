"""Lightweight molecular structure analysis, visualisation, graph export, and coarse-graining.

molscope reads ``.xyz``, ``.pdb``, ``.cif`` and ``.sdf`` files (optionally
gzip-compressed), lets you select and measure atoms, analyse structures and
ensembles, export molecular graphs for machine learning, coarse-grain onto
beads, and visualise everything in 3D.

What it does
------------
- **Read and write** XYZ, PDB, mmCIF and SDF; fetch by id from RCSB; load
  multi-model NMR ensembles (:func:`read`, :func:`fetch`, :func:`read_pdb_models`).
- **Select and measure** by chain, element or residue; distances, angles,
  dihedrals and Kabsch-aligned RMSD (:class:`Molecule`).
- **Analyse** centroids, radius of gyration, inertia tensor, bonds and contacts.
- **Contact maps** at atom or residue level (:class:`ContactMap`).
- **Ensembles**: pairwise RMSD, RMSF, averaging, conformer clustering
  (:mod:`molscope.ensemble`, :func:`cluster`, :func:`rmsd_matrix`).
- **Export for ML**: structural descriptors and molecular graphs for NetworkX,
  PyTorch Geometric and DGL (:func:`descriptors`, :class:`MolecularGraph`).
- **Coarse-grain** onto residue, Martini-style or custom bead mappings
  (:mod:`molscope.coarsegrain`).
- **Visualise** with 3D matplotlib plots, an interactive py3Dmol viewer, and
  spin GIFs.

Examples
--------
>>> import molscope as ms
>>> mol = ms.read("1fqy.pdb")          # parser chosen from the extension
>>> mol = ms.fetch("1fqy")             # ...or download straight from RCSB by id
>>> print(mol.summary())               # atoms, formula, chains, bounding box
>>> ca = mol.alpha_carbons()           # select the C-alpha atoms
>>> mol.plot(color_by="chain")         # render in 3D

>>> g = mol.to_graph()                 # molecular graph for ML, no extra deps
>>> cg = mol.coarse_grain("residue_com")   # one bead per residue

See https://github.com/roshan2004/molscope for the full documentation.
"""

from . import coarsegrain, dssp, ensemble
from .coarsegrain import BeadMapping, BondMapping, CoarseGrainReport, DroppedAtom
from .contactmap import ContactMap
from .descriptors import descriptors, featurize_many
from .dssp import SecondaryStructure
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
    "SecondaryStructure",
    "cluster",
    "coarsegrain",
    "descriptors",
    "dssp",
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
__version__ = "0.7.0"
