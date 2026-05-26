"""Tests for the molecular graph layer and its exporters."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import MolecularGraph, Molecule
from molscope.graph import edge_feature_names, node_feature_names

DATA = os.path.dirname(os.path.dirname(__file__))


def water():
    coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    return Molecule(coords, ["O", "H", "H"], name="water")


# -- core graph (no optional deps) ------------------------------------------


def test_to_graph_nodes_and_edges():
    g = water().to_graph()
    assert isinstance(g, MolecularGraph)
    assert g.n_atoms == 3
    assert g.n_bonds == 2  # the two O-H bonds
    np.testing.assert_array_equal(g.atomic_numbers, [8, 1, 1])
    assert g.masses[0] == pytest.approx(15.999)


def test_graph_edge_distances_match_geometry():
    g = water().to_graph()
    # every edge distance equals the coordinate distance of its endpoints
    for (i, j), d in zip(g.edges, g.edge_distances):
        assert d == pytest.approx(np.linalg.norm(g.coords[i] - g.coords[j]))


def test_node_features_shape():
    feats = water().to_graph().node_features()
    assert feats.shape == (3, 2)  # [atomic_number, mass]


def test_graph_feature_presets_have_stable_names_and_shapes():
    g = water().to_graph()
    x, e, node_names, edge_names = g.feature_matrices(return_names=True)
    assert node_names == node_feature_names("ml")
    assert edge_names == edge_feature_names("ml")
    assert x.shape == (3, len(node_names))
    assert e.shape == (2, len(edge_names))
    assert "element_O" in node_names
    assert "formal_charge" in node_names
    assert "bond_order" in edge_names


def test_graph_basic_feature_presets_include_charge_and_bond_order():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
        ["N", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
        formal_charges=[1, -1],
    )
    g = mol.to_graph()
    x, node_names = g.node_features("basic", return_names=True)
    e, edge_names = g.edge_features("basic", return_names=True)
    assert node_names == ["atomic_number", "mass", "formal_charge"]
    assert edge_names == ["distance", "bond_order"]
    np.testing.assert_array_equal(x[:, node_names.index("formal_charge")], [1.0, -1.0])
    np.testing.assert_array_equal(e[:, edge_names.index("bond_order")], [2.0])


def test_graph_ml_preset_marks_aromatic_bond_order():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        ["C", "C"],
        bond_index=[[0, 1]],
        bond_orders=[1.5],
    )
    e, names = mol.to_graph().edge_features("ml", return_names=True)
    assert e[0, names.index("aromatic")] == 1.0


def test_to_graph_accepts_explicit_bonds():
    g = water().to_graph(bonds=[[0, 1]])
    assert g.n_bonds == 1


def test_to_graph_preserves_explicit_bond_orders():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
        ["C", "C"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    g = mol.to_graph()
    np.testing.assert_array_equal(g.edge_types, [2.0])


def test_graph_carries_metadata():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    g = mol.to_graph()
    assert g.n_atoms == 1661
    assert len(g.chains) == 1661 and g.chains[0] == "A"


def test_graph_carries_formal_charges():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["O", "C"],
        bond_index=[[0, 1]],
        formal_charges=[-1, 1],
    )
    g = mol.to_graph()
    np.testing.assert_array_equal(g.formal_charges, [-1, 1])


def test_graph_can_attach_rdkit_aromatic_features():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([
            [1.396, 0.000, 0.000],
            [0.698, 1.209, 0.000],
            [-0.698, 1.209, 0.000],
            [-1.396, 0.000, 0.000],
            [-0.698, -1.209, 0.000],
            [0.698, -1.209, 0.000],
        ]),
        ["C"] * 6,
        bond_index=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
        bond_orders=[1.5] * 6,
    )
    g = mol.to_graph(include_chemical_features=True)
    assert g.aromatic_atoms.all()
    assert g.aromatic_bonds.all()


# -- networkx (in dev deps, tested for real) --------------------------------


def test_to_networkx():
    nx = pytest.importorskip("networkx")
    G = water().to_networkx()
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.nodes[0]["element"] == "O"
    assert G.nodes[0]["atomic_number"] == 8
    # edge attributes present
    i, j = next(iter(G.edges))
    assert "distance" in G.edges[i, j]


def test_networkx_includes_residue_metadata():
    pytest.importorskip("networkx")
    G = ms.read_pdb(os.path.join(DATA, "1fqy.pdb")).to_networkx()
    assert G.nodes[0]["chain"] == "A"
    assert G.nodes[0]["resname"] == "LYS"


def test_networkx_includes_formal_charge():
    pytest.importorskip("networkx")
    mol = Molecule(np.zeros((1, 3)), ["N"], formal_charges=[1])
    G = mol.to_networkx()
    assert G.nodes[0]["formal_charge"] == 1


# -- PyTorch Geometric / DGL (skipped unless installed) ---------------------


def test_to_pyg_data():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = water().to_pyg_data()
    assert data.num_nodes == 3
    assert data.x.shape == (3, 2)
    assert data.pos.shape == (3, 3)
    # 2 undirected bonds -> 4 directed edges
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, 1)
    assert data.bond_order.shape == (4,)
    assert data.formal_charge.shape == (3,)


def test_to_dgl_graph():
    pytest.importorskip("dgl")
    pytest.importorskip("torch")
    g = water().to_dgl_graph()
    assert g.num_nodes() == 3
    assert g.num_edges() == 4
    assert g.ndata["feat"].shape == (3, 2)
    assert g.ndata["formal_charge"].shape == (3,)
    assert g.edata["bond_order"].shape == (4,)
