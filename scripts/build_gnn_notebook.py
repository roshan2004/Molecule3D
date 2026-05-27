"""Generate notebooks/pdb_to_gnn.ipynb (the structure -> trained GNN demo).

Kept as a script so the notebook JSON is always well-formed and regenerable.
The code cells mirror examples/pdb_to_pyg_ml.py, which is exercised by the
test suite, so the narrative notebook and the tested script cannot drift far.
"""

from __future__ import annotations

import json
from pathlib import Path


def _lines(src):
    text = "\n".join(src)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(*src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(src),
    }


cells = [
    md(
        "# From a PDB structure to a trained GNN",
        "",
        "This notebook walks the path MolScope is built to make easy: read a",
        "structure, turn each model into a molecular graph, export to PyTorch",
        "Geometric, and train a small graph neural network end to end.",
        "",
        "The dataset is the bundled 20-model NMR structure `1aml`. Each model",
        "becomes one graph, and the toy graph-level tasks are:",
        "",
        "- **regress** the radius of gyration, and",
        "- **classify** whether a conformer is more expanded than the median.",
        "",
        "MolScope's core is just NumPy + Matplotlib. The GNN parts need the",
        "optional ML stack:",
        "",
        "```bash",
        "pip install 'molscope[pyg]'   # torch + torch_geometric",
        "```",
    ),
    code(
        "import molscope as ms",
        "",
        "try:",
        "    import torch",
        "    import torch.nn.functional as F",
        "    from torch_geometric.loader import DataLoader",
        "    from torch_geometric.nn import GCNConv, global_mean_pool",
        "except ImportError as exc:",
        "    raise SystemExit(",
        '        \"Install the optional ML stack to run this notebook:\\n\"',
        "        \"  pip install 'molscope[pyg]'\"",
        "    ) from exc",
        "",
        "torch.manual_seed(7)",
        'print(\"molscope\", ms.__version__, \"| torch\", torch.__version__)',
    ),
    md(
        "## 1. Read the NMR ensemble",
        "",
        "`read_pdb_models` returns one `Molecule` per model. No ML dependency is",
        "involved yet; this is plain structure parsing.",
    ),
    code(
        "from pathlib import Path",
        "",
        "# Works whether the notebook runs from the repo root or notebooks/.",
        'DATA = Path(\"examples/data\")',
        "if not DATA.exists():",
        '    DATA = Path(\"..\") / \"examples\" / \"data\"',
        'ENSEMBLE = DATA / \"1aml.pdb\"',
        "",
        "models = ms.read_pdb_models(ENSEMBLE)",
        'print(f\"{len(models)} models\")',
        "print(models[0].summary())",
    ),
    md(
        "## 2. Structure to graph",
        "",
        "`to_graph()` builds a molecular graph with **no extra dependencies**.",
        "`to_pyg_data()` is the PyTorch Geometric hand-off used below. The `ml`",
        "presets select richer node/edge feature sets aimed at learning.",
    ),
    code(
        "g = models[0].to_graph()",
        'print(f\"to_graph(): {g.n_atoms} nodes, {g.n_bonds} bonds\")',
        "",
        'data0 = models[0].to_pyg_data(node_preset=\"ml\", edge_preset=\"ml\")',
        'print(f\"to_pyg_data(): {data0}\")',
    ),
    md(
        "## 3. Build the PyG dataset with graph-level labels",
        "",
        "Each model becomes one `Data` graph. We standardise the regression",
        "target and append centred 3D coordinates to the node features so the",
        "geometry target is learnable in this compact example.",
    ),
    code(
        "radii = torch.tensor([m.radius_of_gyration for m in models], dtype=torch.float)",
        "rg_mean = radii.mean()",
        "rg_std = radii.std().clamp_min(1e-6)",
        "median = radii.median()",
        "",
        "graphs = []",
        "for idx, mol in enumerate(models):",
        '    data = mol.to_pyg_data(node_preset=\"ml\", edge_preset=\"ml\")',
        "    centered_pos = data.pos - data.pos.mean(dim=0, keepdim=True)",
        "    data.x = torch.cat([data.x, centered_pos], dim=1)",
        "    data.y_reg = ((radii[idx] - rg_mean) / rg_std).view(1)",
        "    data.y_cls = (radii[idx] > median).float().view(1)",
        "    data.model_id = idx + 1",
        "    graphs.append(data)",
        "",
        'print(f\"{len(graphs)} graphs, {graphs[0].x.size(1)} node features each\")',
    ),
    md(
        "## 4. Define a small GNN",
        "",
        "Two `GCNConv` layers, mean-pooled to a graph embedding, with separate",
        "regression and classification heads sharing the same backbone.",
    ),
    code(
        "class GraphRegressorClassifier(torch.nn.Module):",
        "    def __init__(self, in_channels):",
        "        super().__init__()",
        "        self.conv1 = GCNConv(in_channels, 64)",
        "        self.conv2 = GCNConv(64, 64)",
        "        self.reg_head = torch.nn.Linear(64, 1)",
        "        self.cls_head = torch.nn.Linear(64, 1)",
        "",
        "    def forward(self, batch):",
        "        x = self.conv1(batch.x, batch.edge_index).relu()",
        "        x = self.conv2(x, batch.edge_index).relu()",
        "        pooled = global_mean_pool(x, batch.batch)",
        "        return self.reg_head(pooled).squeeze(-1), self.cls_head(pooled).squeeze(-1)",
        "",
        "",
        "model = GraphRegressorClassifier(graphs[0].x.size(1))",
        "print(model)",
    ),
    md(
        "## 5. Train",
        "",
        "A 14/6 train/test split over the 20 conformers, optimising the summed",
        "regression (MSE) and classification (BCE) losses.",
    ),
    code(
        "order = torch.randperm(len(graphs), generator=torch.Generator().manual_seed(7)).tolist()",
        "train_graphs = [graphs[i] for i in order[:14]]",
        "test_graphs = [graphs[i] for i in order[14:]]",
        "",
        "train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)",
        "test_loader = DataLoader(test_graphs, batch_size=6)",
        "",
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)",
        "",
        "for epoch in range(1, 81):",
        "    model.train()",
        "    total = 0.0",
        "    for batch in train_loader:",
        "        pred_reg, pred_cls = model(batch)",
        "        loss_reg = F.mse_loss(pred_reg, batch.y_reg.view(-1))",
        "        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, batch.y_cls.view(-1))",
        "        loss = loss_reg + loss_cls",
        "        optimizer.zero_grad()",
        "        loss.backward()",
        "        optimizer.step()",
        "        total += float(loss) * batch.num_graphs",
        "    if epoch in {1, 20, 40, 80}:",
        '        print(f\"epoch {epoch:02d} train_loss={total / len(train_graphs):.3f}\")',
    ),
    md(
        "## 6. Evaluate on the held-out conformers",
        "",
        "Predictions are de-standardised back to angstroms for the radius of",
        "gyration, alongside the expanded/constrained classification accuracy.",
    ),
    code(
        "model.eval()",
        "with torch.no_grad():",
        "    batch = next(iter(test_loader))",
        "    pred_reg, pred_cls = model(batch)",
        "    pred_rg = pred_reg * rg_std + rg_mean",
        "    true_rg = batch.y_reg.view(-1) * rg_std + rg_mean",
        "    mae = (pred_rg - true_rg).abs().mean()",
        "    acc = ((pred_cls.sigmoid() > 0.5) == batch.y_cls.view(-1).bool()).float().mean()",
        "",
        'print(f\"radius-of-gyration MAE: {mae:.3f} A\")',
        'print(f\"expanded/constrained accuracy: {acc:.2%}\")',
        'print()',
        "for model_id, predicted, observed in zip(batch.model_id.tolist(), pred_rg, true_rg):",
        (
            '    print(f\"  model {model_id:02d}: predicted '
            '{predicted:.2f} A, observed {observed:.2f} A\")'
        ),
    ),
    md(
        "## Where to go next",
        "",
        "- Swap `1aml` for your own structure with `ms.read(...)` or",
        "  `ms.fetch('<pdb id>')`.",
        "- Inspect the feature columns with `ms.node_feature_names('ml')` and",
        "  `ms.edge_feature_names('ml')`.",
        "- Use `mol.to_dgl_graph()` instead for a DGL pipeline, or",
        "  `mol.to_networkx()` for classical graph analysis.",
        "",
        "The script form of this walkthrough is",
        "[`examples/pdb_to_pyg_ml.py`](../examples/pdb_to_pyg_ml.py).",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "pdb_to_gnn.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {out} ({len(cells)} cells)")
