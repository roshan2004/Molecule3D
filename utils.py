"""Backward-compatible shim for the original ``utils`` API.

The implementation now lives in the :mod:`molecule3d` package. This module
keeps the old ``Molecule`` / ``Pdb`` classes working so existing scripts and
notebooks don't break, but new code should use ``molecule3d`` directly::

    import molecule3d as m3d
    m3d.read("helix_201.xyz").translate((1, 2, -1)).plot()
"""

from molecule3d import io


class Molecule:
    """Deprecated. Use ``molecule3d.read`` / ``molecule3d.Molecule`` instead."""

    _reader = staticmethod(io.read_xyz)

    def __init__(self, file):
        self.file = file
        self.coords = []
        self._mol = None

    def coordinates(self):
        self._mol = self._reader(self.file)
        self.coords = [tuple(row) for row in self._mol.coords]
        return self.coords

    def translate(self, t):
        self._mol = self._mol.translate(t)
        self.coords = [tuple(row) for row in self._mol.coords]
        return self.coords

    def graph(self):
        self._mol.plot()


class Pdb(Molecule):
    """Deprecated. Use ``molecule3d.read_pdb`` instead."""

    _reader = staticmethod(io.read_pdb)


def main():
    molecule = Molecule("helix_201.xyz")
    molecule.coordinates()
    molecule.translate((1, 2, -1))
    molecule.graph()


if __name__ == "__main__":
    main()
