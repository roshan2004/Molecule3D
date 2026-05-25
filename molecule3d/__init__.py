"""molecule3d: read molecular coordinate files and plot atoms in 3D."""

from .molecule import Molecule
from .io import read, read_pdb, read_xyz

__all__ = ["Molecule", "read", "read_pdb", "read_xyz"]
__version__ = "0.1.0"
