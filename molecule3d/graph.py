"""Molecular graph layer.

``Molecule.to_graph()`` turns 3D coordinates plus inferred bonds into a
:class:`MolecularGraph` — a small, dependency-free container of node and edge
data. From there it exports to the common ML graph formats:

    G   = mol.to_graph()
    nxg = mol.to_networkx()     # networkx.Graph
    data = mol.to_pyg_data()    # torch_geometric.data.Data
    dglg = mol.to_dgl_graph()   # dgl.DGLGraph

The exporters import their backend lazily, so networkx / PyTorch Geometric / DGL
are only required if you actually call the matching method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import elements


@dataclass
class MolecularGraph:
    """Atoms as nodes, bonds as edges, with attributes attached to both.

    Nodes carry element, atomic number, mass, coordinates and (when available)
    atom name, residue name/id and chain. Edges carry the bonded atom pair, the
    interatomic distance, and a bond order (``1.0`` for geometrically inferred
    bonds, whose order is unknown).
    """

    coords: np.ndarray              # (N, 3)
    elements: list[str]
    edges: np.ndarray               # (E, 2), i < j
    edge_distances: np.ndarray      # (E,)
    edge_types: np.ndarray          # (E,) bond order; 1.0 when inferred/unknown
    atom_names: list[str] = field(default_factory=list)
    resnames: list[str] = field(default_factory=list)
    resids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    chains: list[str] = field(default_factory=list)
    name: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def n_bonds(self) -> int:
        return len(self.edges)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.array([elements.atomic_number(e) for e in self.elements])

    @property
    def masses(self) -> np.ndarray:
        return np.array([elements.mass(e) for e in self.elements])

    def node_features(self) -> np.ndarray:
        """Default ``(N, 2)`` node feature matrix: ``[atomic_number, mass]``."""
        return np.stack([self.atomic_numbers, self.masses], axis=1).astype(float)

    # -- exporters ----------------------------------------------------------

    def to_networkx(self):
        """Return a ``networkx.Graph`` with node and edge attributes."""
        nx = _require("networkx", "networkx", "pip install networkx")
        g = nx.Graph(name=self.name)
        z, m = self.atomic_numbers, self.masses
        for i in range(self.n_atoms):
            attrs = {
                "element": self.elements[i],
                "atomic_number": int(z[i]),
                "mass": float(m[i]),
                "pos": tuple(float(c) for c in self.coords[i]),
            }
            if self.atom_names:
                attrs["atom_name"] = self.atom_names[i]
            if self.resnames:
                attrs["resname"] = self.resnames[i]
            if len(self.resids):
                attrs["resid"] = int(self.resids[i])
            if self.chains:
                attrs["chain"] = self.chains[i]
            g.add_node(i, **attrs)
        for (i, j), dist, btype in zip(self.edges, self.edge_distances, self.edge_types):
            g.add_edge(int(i), int(j), distance=float(dist), bond_type=float(btype))
        return g

    def to_pyg_data(self):
        """Return a ``torch_geometric.data.Data`` object.

        Populates ``x`` (node features), ``z`` (atomic numbers), ``pos`` (3D
        coordinates), ``edge_index`` and ``edge_attr`` (distances). Edges are
        made bidirectional for message passing.
        """
        torch = _require("torch", "PyTorch Geometric", "pip install torch torch_geometric")
        Data = _require(
            "torch_geometric.data", "PyTorch Geometric",
            "pip install torch torch_geometric", attr="Data",
        )
        src, dst, dist = self._directed_edges()
        return Data(
            x=torch.tensor(self.node_features(), dtype=torch.float),
            z=torch.tensor(self.atomic_numbers, dtype=torch.long),
            pos=torch.tensor(self.coords, dtype=torch.float),
            edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
            edge_attr=torch.tensor(dist[:, None], dtype=torch.float),
            num_nodes=self.n_atoms,
        )

    def to_dgl_graph(self):
        """Return a ``dgl.DGLGraph`` with node/edge feature tensors."""
        dgl = _require("dgl", "DGL", "pip install dgl")
        torch = _require("torch", "DGL", "pip install dgl torch")
        src, dst, dist = self._directed_edges()
        g = dgl.graph(
            (torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
            num_nodes=self.n_atoms,
        )
        g.ndata["feat"] = torch.tensor(self.node_features(), dtype=torch.float)
        g.ndata["z"] = torch.tensor(self.atomic_numbers, dtype=torch.long)
        g.ndata["pos"] = torch.tensor(self.coords, dtype=torch.float)
        g.edata["distance"] = torch.tensor(dist, dtype=torch.float)
        return g

    def _directed_edges(self):
        """Edges in both directions: (src, dst, distance) for message passing."""
        if self.n_bonds == 0:
            empty_i = np.empty(0, dtype=int)
            return empty_i, empty_i, np.empty(0, dtype=float)
        i, j = self.edges[:, 0], self.edges[:, 1]
        src = np.concatenate([i, j])
        dst = np.concatenate([j, i])
        dist = np.concatenate([self.edge_distances, self.edge_distances])
        return src, dst, dist


def _require(module: str, feature: str, hint: str, attr: Optional[str] = None):
    """Import a backend module (or attribute), raising a friendly error if absent."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(f"{feature} is required for this export; {hint}") from exc
    return getattr(mod, attr) if attr else mod
