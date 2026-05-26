"""Molecular graph layer.

``Molecule.to_graph()`` turns 3D coordinates plus explicit or inferred bonds
into a :class:`MolecularGraph` — a small, dependency-free container of node and edge
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

DEFAULT_GRAPH_ELEMENTS = (
    "H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "NA", "MG", "CA", "FE", "ZN",
)

GRAPH_NODE_FEATURE_PRESETS = ("default", "basic", "ml")
GRAPH_EDGE_FEATURE_PRESETS = ("default", "basic", "ml")


@dataclass
class MolecularGraph:
    """Atoms as nodes, bonds as edges, with attributes attached to both.

    Nodes carry element, atomic number, mass, coordinates and (when available)
    atom name, residue name/id, chain and formal charge. Edges carry the bonded
    atom pair, the interatomic distance, and a bond order (``1.0`` for inferred
    or unknown orders).
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
    formal_charges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    aromatic_atoms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    aromatic_bonds: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

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

    def node_features(
        self,
        preset: str = "default",
        *,
        elements_to_encode=DEFAULT_GRAPH_ELEMENTS,
        return_names: bool = False,
    ):
        """Return a node feature matrix for a named preset.

        ``"default"`` preserves the historical ``[atomic_number, mass]`` matrix.
        ``"basic"`` adds formal charge. ``"ml"`` adds a fixed element one-hot
        basis and an aromatic-atom flag, using zeros when aromaticity was not
        supplied by optional chemical perception.
        """
        names = node_feature_names(preset, elements_to_encode=elements_to_encode)
        arrays = []
        for name in names:
            if name == "atomic_number":
                arrays.append(self.atomic_numbers.astype(float))
            elif name == "mass":
                arrays.append(self.masses.astype(float))
            elif name == "formal_charge":
                arrays.append(self._formal_charges_or_zero().astype(float))
            elif name == "aromatic":
                arrays.append(self._aromatic_atoms_or_false().astype(float))
            elif name.startswith("element_"):
                symbol = name.removeprefix("element_").upper()
                arrays.append(np.array(
                    [1.0 if (element or "").upper() == symbol else 0.0
                     for element in self.elements],
                    dtype=float,
                ))
            else:  # pragma: no cover - protected by node_feature_names
                raise ValueError(f"unknown node feature {name!r}")
        matrix = np.stack(arrays, axis=1).astype(float) if arrays else np.empty((self.n_atoms, 0))
        return (matrix, names) if return_names else matrix

    def edge_features(self, preset: str = "default", *, return_names: bool = False):
        """Return an edge feature matrix for a named preset.

        ``"default"`` is distance-only for backwards compatibility. ``"basic"``
        adds bond order. ``"ml"`` adds an aromatic-bond flag, using explicit
        aromaticity when available and otherwise treating order ``1.5`` as
        aromatic.
        """
        names = edge_feature_names(preset)
        arrays = []
        for name in names:
            if name == "distance":
                arrays.append(np.asarray(self.edge_distances, dtype=float))
            elif name == "bond_order":
                arrays.append(np.asarray(self.edge_types, dtype=float))
            elif name == "aromatic":
                arrays.append(self._aromatic_bonds_or_order().astype(float))
            else:  # pragma: no cover - protected by edge_feature_names
                raise ValueError(f"unknown edge feature {name!r}")
        matrix = np.stack(arrays, axis=1).astype(float) if arrays else np.empty((self.n_bonds, 0))
        return (matrix, names) if return_names else matrix

    def feature_matrices(
        self,
        *,
        node_preset: str = "ml",
        edge_preset: str = "ml",
        elements_to_encode=DEFAULT_GRAPH_ELEMENTS,
        return_names: bool = False,
    ):
        """Return ``(node_features, edge_features)`` for graph ML workflows."""
        x, node_names = self.node_features(
            node_preset,
            elements_to_encode=elements_to_encode,
            return_names=True,
        )
        e, edge_names = self.edge_features(edge_preset, return_names=True)
        if return_names:
            return x, e, node_names, edge_names
        return x, e

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
            if len(self.formal_charges):
                attrs["formal_charge"] = int(self.formal_charges[i])
            if len(self.aromatic_atoms):
                attrs["aromatic"] = bool(self.aromatic_atoms[i])
            g.add_node(i, **attrs)
        aromatic = self._aromatic_bonds_or_order()
        for edge_idx, ((i, j), dist, btype) in enumerate(
            zip(self.edges, self.edge_distances, self.edge_types)
        ):
            g.add_edge(
                int(i), int(j), distance=float(dist), bond_type=float(btype),
                aromatic=bool(aromatic[edge_idx]),
            )
        return g

    def to_pyg_data(self, node_preset: str = "default", edge_preset: str = "default"):
        """Return a ``torch_geometric.data.Data`` object.

        Populates ``x`` (node features), ``z`` (atomic numbers), ``pos`` (3D
        coordinates), ``edge_index`` and ``edge_attr``. Edges are made
        bidirectional for message passing.
        """
        torch = _require("torch", "PyTorch Geometric", "pip install torch torch_geometric")
        Data = _require(
            "torch_geometric.data", "PyTorch Geometric",
            "pip install torch torch_geometric", attr="Data",
        )
        src, dst, dist, order = self._directed_edges()
        return Data(
            x=torch.tensor(self.node_features(node_preset), dtype=torch.float),
            node_feature_names=node_feature_names(node_preset),
            z=torch.tensor(self.atomic_numbers, dtype=torch.long),
            formal_charge=torch.tensor(self._formal_charges_or_zero(), dtype=torch.long),
            pos=torch.tensor(self.coords, dtype=torch.float),
            edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
            edge_attr=torch.tensor(self._directed_edge_features(edge_preset), dtype=torch.float),
            edge_feature_names=edge_feature_names(edge_preset),
            bond_order=torch.tensor(order, dtype=torch.float),
            num_nodes=self.n_atoms,
        )

    def to_dgl_graph(self, node_preset: str = "default", edge_preset: str = "default"):
        """Return a ``dgl.DGLGraph`` with node/edge feature tensors."""
        dgl = _require("dgl", "DGL", "pip install dgl")
        torch = _require("torch", "DGL", "pip install dgl torch")
        src, dst, dist, order = self._directed_edges()
        g = dgl.graph(
            (torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
            num_nodes=self.n_atoms,
        )
        g.ndata["feat"] = torch.tensor(self.node_features(node_preset), dtype=torch.float)
        g.ndata["z"] = torch.tensor(self.atomic_numbers, dtype=torch.long)
        g.ndata["formal_charge"] = torch.tensor(self._formal_charges_or_zero(), dtype=torch.long)
        g.ndata["pos"] = torch.tensor(self.coords, dtype=torch.float)
        g.edata["feat"] = torch.tensor(self._directed_edge_features(edge_preset), dtype=torch.float)
        g.edata["distance"] = torch.tensor(dist, dtype=torch.float)
        g.edata["bond_order"] = torch.tensor(order, dtype=torch.float)
        return g

    def _directed_edges(self):
        """Edges in both directions: (src, dst, distance) for message passing."""
        if self.n_bonds == 0:
            empty_i = np.empty(0, dtype=int)
            empty_f = np.empty(0, dtype=float)
            return empty_i, empty_i, empty_f, empty_f
        i, j = self.edges[:, 0], self.edges[:, 1]
        src = np.concatenate([i, j])
        dst = np.concatenate([j, i])
        dist = np.concatenate([self.edge_distances, self.edge_distances])
        order = np.concatenate([self.edge_types, self.edge_types])
        return src, dst, dist, order

    def _formal_charges_or_zero(self) -> np.ndarray:
        if len(self.formal_charges):
            return np.asarray(self.formal_charges, dtype=int)
        return np.zeros(self.n_atoms, dtype=int)

    def _aromatic_atoms_or_false(self) -> np.ndarray:
        if len(self.aromatic_atoms):
            return np.asarray(self.aromatic_atoms, dtype=bool)
        return np.zeros(self.n_atoms, dtype=bool)

    def _aromatic_bonds_or_order(self) -> np.ndarray:
        if len(self.aromatic_bonds):
            return np.asarray(self.aromatic_bonds, dtype=bool)
        return np.isclose(np.asarray(self.edge_types, dtype=float), 1.5)

    def _directed_edge_features(self, preset: str) -> np.ndarray:
        undirected = self.edge_features(preset)
        if self.n_bonds == 0:
            return undirected
        return np.concatenate([undirected, undirected], axis=0)


def node_feature_names(preset: str = "default", *, elements_to_encode=DEFAULT_GRAPH_ELEMENTS):
    """Feature names returned by :meth:`MolecularGraph.node_features`."""
    preset = _normalize_preset(preset, GRAPH_NODE_FEATURE_PRESETS, "node")
    if preset == "default":
        return ["atomic_number", "mass"]
    if preset == "basic":
        return ["atomic_number", "mass", "formal_charge"]
    return (
        [f"element_{symbol.upper()}" for symbol in elements_to_encode]
        + ["atomic_number", "mass", "formal_charge", "aromatic"]
    )


def edge_feature_names(preset: str = "default"):
    """Feature names returned by :meth:`MolecularGraph.edge_features`."""
    preset = _normalize_preset(preset, GRAPH_EDGE_FEATURE_PRESETS, "edge")
    if preset == "default":
        return ["distance"]
    if preset == "basic":
        return ["distance", "bond_order"]
    return ["distance", "bond_order", "aromatic"]


def _normalize_preset(preset: str, allowed: tuple[str, ...], label: str) -> str:
    if preset not in allowed:
        choices = "', '".join(allowed)
        raise ValueError(f"unknown {label} feature preset {preset!r}; expected '{choices}'")
    return preset


def _require(module: str, feature: str, hint: str, attr: Optional[str] = None):
    """Import a backend module (or attribute), raising a friendly error if absent."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(f"{feature} is required for this export; {hint}") from exc
    return getattr(mod, attr) if attr else mod
