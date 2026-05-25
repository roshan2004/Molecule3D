"""Tests for the molecular graph layer and its exporters."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import MolecularGraph, Molecule

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


def test_to_graph_accepts_explicit_bonds():
    g = water().to_graph(bonds=[[0, 1]])
    assert g.n_bonds == 1


def test_graph_carries_metadata():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    g = mol.to_graph()
    assert g.n_atoms == 1661
    assert len(g.chains) == 1661 and g.chains[0] == "A"


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


def test_to_dgl_graph():
    pytest.importorskip("dgl")
    pytest.importorskip("torch")
    g = water().to_dgl_graph()
    assert g.num_nodes() == 3
    assert g.num_edges() == 4
    assert g.ndata["feat"].shape == (3, 2)
