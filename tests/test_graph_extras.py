from unittest.mock import patch

import numpy as np
import pytest

from molscope import MolecularGraph, Molecule
from molscope.graph import ResidueContactGraph

# Patch target that makes the scipy import fail, forcing the dense fallback.
_NO_SCIPY = {"scipy": None, "scipy.sparse": None, "scipy.sparse.linalg": None}


def test_geometric_edge_features_water():
    # Water molecule: oxygen at index 0, hydrogens at 1 and 2
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ])
    mol = Molecule(coords, ["O", "H", "H"], bond_index=[[0, 1], [0, 2]], bond_orders=[1, 1])
    g = mol.to_graph()
    
    # Feature preset check
    feats, names = g.edge_features("geom", return_names=True)
    assert names == ["distance", "bond_order", "aromatic", "bond_angle", "dihedral"]
    
    # Distance checks
    np.testing.assert_allclose(feats[0, 0], mol.distance(0, 1))
    np.testing.assert_allclose(feats[1, 0], mol.distance(0, 2))
    
    # Bond order checks
    assert feats[0, 1] == 1.0
    assert feats[1, 1] == 1.0
    
    # Aromaticity checks
    assert feats[0, 2] == 0.0
    assert feats[1, 2] == 0.0
    
    # Angle check: the angle at oxygen (2-0-1) is ~104.5 degrees
    expected_angle = mol.angle(1, 0, 2)
    np.testing.assert_allclose(feats[0, 3], expected_angle, rtol=1e-5)
    np.testing.assert_allclose(feats[1, 3], expected_angle, rtol=1e-5)
    
    # Dihedral check: should be 0.0 since no 4-body torsions exist
    assert feats[0, 4] == 0.0
    assert feats[1, 4] == 0.0


def test_geometric_edge_features_butane_dihedral():
    # 4-atom carbon chain (butane-like backbone)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.2, 0.0],
        [3.5, 1.2, 0.5]
    ])
    mol = Molecule(coords, ["C", "C", "C", "C"], bond_index=[[0, 1], [1, 2], [2, 3]])
    g = mol.to_graph()
    
    feats = g.edge_features("geom")
    
    # Bonds are [0, 1], [1, 2], [2, 3]
    # For [1, 2] (index 1), the dihedral is 0-1-2-3
    expected_dihedral = mol.dihedral(0, 1, 2, 3)
    np.testing.assert_allclose(feats[1, 4], expected_dihedral, rtol=1e-5)
    
    # Dihedrals for end bonds [0, 1] and [2, 3] should be 0.0
    assert feats[0, 4] == 0.0
    assert feats[2, 4] == 0.0


def test_geometric_features_empty_graph():
    coords = np.zeros((0, 3))
    g = MolecularGraph(
        coords=coords,
        elements=[],
        edges=np.empty((0, 2), dtype=int),
        edge_distances=np.empty(0),
        edge_types=np.empty(0)
    )
    feats = g.edge_features("geom")
    assert feats.shape == (0, 5)


def test_laplacian_pe_dense_vs_sparse():
    # Create a small ring graph (benzene-like connectivity) to have n_atoms > k + 1
    # For k=3, we need n_atoms > 4
    coords = np.random.rand(6, 3)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    g = MolecularGraph(
        coords=coords,
        elements=["C"] * 6,
        edges=edges,
        edge_distances=np.ones(5),
        edge_types=np.ones(5)
    )
    
    # Force dense execution by patching out scipy
    import sys
    with patch.dict(sys.modules, _NO_SCIPY):
        pe_dense = g.laplacian_pe(k=3)
        
    # Execute normally (using SciPy if available)
    pe_sparse = g.laplacian_pe(k=3)

    # Eigenvectors are unique up to sign, but _sign_stabilize() fixes the sign
    # deterministically, so the two backends must agree on signed values too.
    np.testing.assert_allclose(pe_dense, pe_sparse, atol=1e-7)


def test_laplacian_pe_dense_vs_sparse_symmetric_molecule():
    # Regression test: on symmetric/larger graphs ARPACK's which="SM" mode used
    # to skip the trivial zero eigenvalue, so the sparse path returned a
    # different eigenvector set than the dense path. Shift-invert (sigma=0) plus
    # deterministic sign stabilization must make the two paths agree exactly.
    # Naphthalene skeleton (fused 6-6 rings) is symmetric and has degeneracies.
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],   # ring A
        [4, 6], [6, 7], [7, 8], [8, 9], [9, 5],            # ring B fused on 4-5
    ])
    n = 10
    coords = np.zeros((n, 3))  # coords are irrelevant to the Laplacian
    g = MolecularGraph(
        coords=coords,
        elements=["C"] * n,
        edges=edges,
        edge_distances=np.ones(len(edges)),
        edge_types=np.ones(len(edges)),
    )

    import sys
    with patch.dict(sys.modules, _NO_SCIPY):
        pe_dense = g.laplacian_pe(k=5)
    pe_sparse = g.laplacian_pe(k=5)

    # Eigenvalues (hence |eigenvectors|) must match...
    np.testing.assert_allclose(np.abs(pe_dense), np.abs(pe_sparse), atol=1e-7)
    # ...and the stabilized signs must match for non-degenerate columns.
    np.testing.assert_allclose(pe_dense, pe_sparse, atol=1e-7)


def test_random_walk_pe_dense_vs_sparse():
    coords = np.random.rand(6, 3)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    g = MolecularGraph(
        coords=coords,
        elements=["C"] * 6,
        edges=edges,
        edge_distances=np.ones(6),
        edge_types=np.ones(6)
    )
    
    import sys
    with patch.dict(sys.modules, _NO_SCIPY):
        pe_dense = g.random_walk_pe(k=5)
        
    pe_sparse = g.random_walk_pe(k=5)
    np.testing.assert_allclose(pe_dense, pe_sparse, atol=1e-10)


def test_residue_contact_graph_pe_dense_vs_sparse():
    coords = np.random.rand(6, 3)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    g = ResidueContactGraph(
        coords=coords,
        edges=edges,
        edge_distances=np.ones(5),
        edge_types=["ca"] * 5,
        resnames=["ALA"] * 6,
        resids=np.arange(6),
        icodes=[""] * 6,
        chains=["A"] * 6,
        residue_sizes=np.ones(6, dtype=int)
    )
    
    import sys
    with patch.dict(sys.modules, _NO_SCIPY):
        lap_dense = g.laplacian_pe(k=3)
        rw_dense = g.random_walk_pe(k=4)
        
    lap_sparse = g.laplacian_pe(k=3)
    rw_sparse = g.random_walk_pe(k=4)

    np.testing.assert_allclose(lap_dense, lap_sparse, atol=1e-7)
    np.testing.assert_allclose(rw_dense, rw_sparse, atol=1e-10)


def test_laplacian_pe_too_small_fallback():
    # If self.n_atoms <= k + 1, it should fall back to dense and not crash
    coords = np.random.rand(4, 3)
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    g = MolecularGraph(
        coords=coords,
        elements=["C"] * 4,
        edges=edges,
        edge_distances=np.ones(3),
        edge_types=np.ones(3)
    )
    # n_atoms = 4, k = 3. k + 1 = 4. n_atoms is not > 4, so it falls back to dense.
    pe = g.laplacian_pe(k=3)
    assert pe.shape == (4, 3)


def test_ml_exporters_propagate_geom_features():
    # Skip unless PyG or DGL is installed (similar to tests in test_graph.py)
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ])
    mol = Molecule(coords, ["O", "H", "H"], bond_index=[[0, 1], [0, 2]], bond_orders=[1, 1])
    data = mol.to_pyg_data(edge_preset="geom")
    
    # 2 undirected bonds -> 4 directed edges
    assert data.edge_attr.shape == (4, 5)
    assert data.edge_feature_names == [
        "distance", "bond_order", "aromatic", "bond_angle", "dihedral",
    ]
