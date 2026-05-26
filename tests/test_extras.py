from pathlib import Path

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]


def test_optional_extras_are_declared_for_supported_backends():
    pyproject = (ROOT / "pyproject.toml").read_text()
    expected = {
        "graph": ["networkx"],
        "chem": ["rdkit"],
        "cif": ["gemmi"],
        "gpu": ["torch"],
        "pyg": ["torch", "torch-geometric"],
        "dgl": ["torch", "dgl"],
        "gnn": ["networkx", "torch", "torch-geometric", "dgl"],
    }
    for extra, dependencies in expected.items():
        block_start = pyproject.index(f"{extra} = [")
        block = pyproject[block_start:pyproject.index("]", block_start)]
        for dependency in dependencies:
            assert dependency in block


def test_public_feature_name_helpers_are_available():
    assert "n_atoms" in ms.descriptor_feature_names("native-basic")
    assert "element_C" in ms.node_feature_names("ml")
    assert ms.edge_feature_names("ml") == ["distance", "bond_order", "aromatic"]
