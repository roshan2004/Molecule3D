"""PDB -> molecular graph -> GNN, end to end.

Shows the path this package is meant to make easy: read a structure, turn it
into a graph, export to PyTorch Geometric, and run a real GNN forward pass.

Run it:

    uv pip install torch torch_geometric     # one-time, optional ML backends
    .venv/bin/python examples/graph_to_gnn.py

Use ``.venv/bin/python`` directly (not ``uv run``), because ``uv run`` re-syncs
to the lockfile and prunes torch/torch_geometric, which are intentionally not
locked dependencies. The core graph and networkx parts run without torch.
"""

from pathlib import Path

import molscope as ms

DATA = Path(__file__).resolve().parent / "data"
STRUCTURE = DATA / "1fqy.pdb"


def main():
    mol = ms.read(str(STRUCTURE))
    print(f"Loaded {mol.summary()}\n")

    # 1. Dependency-free graph layer.
    g = mol.to_graph()
    print(f"to_graph(): {g.n_atoms} nodes, {g.n_bonds} bonds")
    print(f"  node features {g.node_features().shape} = [atomic_number, mass]")

    # 2. networkx (optional: pip install 'molscope[graph]').
    try:
        G = mol.to_networkx()
        print(f"to_networkx(): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  node 0: {G.nodes[0]}")
    except ImportError:
        print("to_networkx(): install networkx  ->  pip install 'molscope[graph]'")

    # 3. PyTorch Geometric + a real GNN forward pass (optional).
    try:
        import torch
        from torch_geometric.nn import GCNConv, global_mean_pool
    except ImportError:
        print("\nto_pyg_data() + GNN: install torch torch_geometric to run this part.")
        return

    data = mol.to_pyg_data()
    print(f"\nto_pyg_data(): {data}")

    conv1 = GCNConv(data.x.size(1), 16)
    conv2 = GCNConv(16, 8)
    h = conv1(data.x, data.edge_index).relu()
    h = conv2(h, data.edge_index).relu()
    embedding = global_mean_pool(h, batch=torch.zeros(data.num_nodes, dtype=torch.long))

    print(f"  per-atom embeddings: {tuple(h.shape)}")
    print(f"  pooled graph embedding: {embedding.detach().numpy().round(3)}")
    print("\nA real GNN forward pass ran on the molecule.")


if __name__ == "__main__":
    main()
