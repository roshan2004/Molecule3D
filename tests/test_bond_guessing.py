import numpy as np

from molscope import Molecule


def test_bond_guessing_ethylene():
    # Ethylene (C2H4)
    # 0, 1: C
    # 2, 3: H (bonded to 0)
    # 4, 5: H (bonded to 1)
    coords = np.array([
        [-0.67, 0.0, 0.0],  # C
        [0.67, 0.0, 0.0],   # C
        [-1.23, 0.93, 0.0], # H
        [-1.23, -0.93, 0.0],# H
        [1.23, 0.93, 0.0],  # H
        [1.23, -0.93, 0.0], # H
    ])
    elements = ["C", "C", "H", "H", "H", "H"]
    
    # Test 1: Using coordinates to infer bonds and orders
    mol = Molecule(coords, elements)
    
    # By default, inferred bond orders are all 1.0
    orders_default = mol.bond_order_array()
    assert np.allclose(orders_default, 1.0)
    
    # With infer_orders=True
    orders_inferred = mol.bond_order_array(infer_orders=True)
    # The C-C bond (usually first or last depending on index ordering, but let's locate it)
    bonds = mol.bonds()
    c_c_idx = -1
    for idx, (u, v) in enumerate(bonds):
        if elements[u] == "C" and elements[v] == "C":
            c_c_idx = idx
            break
            
    assert c_c_idx != -1
    assert orders_inferred[c_c_idx] == 2.0  # Double bond guessed
    # All C-H bonds should be 1.0
    for idx in range(len(bonds)):
        if idx != c_c_idx:
            assert orders_inferred[idx] == 1.0

    # Test 2: Using explicit bonds (with bond_orders = None)
    mol_explicit = Molecule(coords, elements, bond_index=bonds)
    assert np.allclose(mol_explicit.bond_order_array(infer_orders=False), 1.0)
    
    orders_explicit = mol_explicit.bond_order_array(infer_orders=True)
    assert orders_explicit[c_c_idx] == 2.0


def test_bond_guessing_acetylene():
    # Acetylene (C2H2)
    # 0, 1: C
    # 2: H (bonded to 0)
    # 3: H (bonded to 1)
    coords = np.array([
        [-0.60, 0.0, 0.0],  # C
        [0.60, 0.0, 0.0],   # C
        [-1.66, 0.0, 0.0],  # H
        [1.66, 0.0, 0.0],   # H
    ])
    elements = ["C", "C", "H", "H"]
    
    mol = Molecule(coords, elements)
    bonds = mol.bonds()
    c_c_idx = -1
    for idx, (u, v) in enumerate(bonds):
        if elements[u] == "C" and elements[v] == "C":
            c_c_idx = idx
            break
            
    assert c_c_idx != -1
    orders = mol.bond_order_array(infer_orders=True)
    assert orders[c_c_idx] == 3.0  # Triple bond guessed
    
    for idx in range(len(bonds)):
        if idx != c_c_idx:
            assert orders[idx] == 1.0


def test_bond_guessing_formaldehyde():
    # Formaldehyde (CH2O)
    # 0: C
    # 1: O
    # 2, 3: H
    coords = np.array([
        [0.0, 0.0, 0.0],    # C
        [1.20, 0.0, 0.0],   # O
        [-0.60, 0.93, 0.0], # H
        [-0.60, -0.93, 0.0],# H
    ])
    elements = ["C", "O", "H", "H"]
    
    mol = Molecule(coords, elements)
    bonds = mol.bonds()
    c_o_idx = -1
    for idx, (u, v) in enumerate(bonds):
        pair = {elements[u], elements[v]}
        if pair == {"C", "O"}:
            c_o_idx = idx
            break
            
    assert c_o_idx != -1
    orders = mol.bond_order_array(infer_orders=True)
    assert orders[c_o_idx] == 2.0  # Double bond guessed
    
    for idx in range(len(bonds)):
        if idx != c_o_idx:
            assert orders[idx] == 1.0


def test_bond_guessing_benzene():
    # Benzene (C6H6)
    # Simplest structure to verify Kekulization
    # C-C ring bonds should alternate 1.0 and 2.0
    elements = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
    bonds = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), # Ring
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11) # C-H
    ]
    coords = np.zeros((12, 3))  # Dummy coordinates as we pass explicit bonds
    
    mol = Molecule(coords, elements, bond_index=bonds)
    orders = mol.bond_order_array(infer_orders=True)
    
    # Ring bonds (first 6) should alternate 1.0 and 2.0
    ring_orders = orders[:6]
    assert sorted(ring_orders) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    
    # C-H bonds should be 1.0
    assert np.allclose(orders[6:], 1.0)


def test_to_graph_infer_orders():
    # Verify that to_graph passes infer_orders correctly
    coords = np.array([
        [-0.67, 0.0, 0.0],
        [0.67, 0.0, 0.0],
        [-1.23, 0.93, 0.0],
        [-1.23, -0.93, 0.0],
        [1.23, 0.93, 0.0],
        [1.23, -0.93, 0.0],
    ])
    elements = ["C", "C", "H", "H", "H", "H"]
    mol = Molecule(coords, elements)
    
    # Graph with default orders
    g_default = mol.to_graph(infer_orders=False)
    assert np.allclose(g_default.edge_types, 1.0)
    
    # Graph with inferred orders
    g_inferred = mol.to_graph(infer_orders=True)
    assert 2.0 in g_inferred.edge_types  # Guessed ethylene double bond
