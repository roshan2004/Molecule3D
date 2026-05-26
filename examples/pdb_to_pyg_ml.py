"""PDB -> graph -> PyTorch Geometric toy classifier/regressor.

This script turns the bundled 20-model NMR PDB structure into a tiny graph
dataset. Each model becomes one molecular graph. The toy ML task is:

* regress radius of gyration
* classify whether a conformer is more expanded than the median conformer

Install the optional ML stack first:

    uv pip install torch torch_geometric
    .venv/bin/python examples/pdb_to_pyg_ml.py

Use ``.venv/bin/python`` directly because ``uv run`` may re-sync the locked
environment and remove optional packages that are not core MolScope deps.
"""

from __future__ import annotations

from pathlib import Path

import molscope as ms

DATA = Path(__file__).resolve().parent / "data"
ENSEMBLE = DATA / "1aml.pdb"


def _require_pyg():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GCNConv, global_mean_pool
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "Install the optional ML stack first:\n"
            "  uv pip install torch torch_geometric\n"
            "  .venv/bin/python examples/pdb_to_pyg_ml.py"
        ) from exc
    return torch, F, DataLoader, GCNConv, global_mean_pool


def build_dataset(torch):
    """Convert NMR models to PyG graphs with toy graph-level labels."""
    models = ms.read_pdb_models(ENSEMBLE)
    radii = torch.tensor([m.radius_of_gyration for m in models], dtype=torch.float)
    rg_mean = radii.mean()
    rg_std = radii.std().clamp_min(1e-6)
    median = radii.median()

    graphs = []
    for idx, mol in enumerate(models):
        data = mol.to_pyg_data(node_preset="ml", edge_preset="ml")

        # GCNConv consumes node features and edge_index. Appending centered 3D
        # coordinates makes this geometry target learnable in a compact example.
        centered_pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        data.x = torch.cat([data.x, centered_pos], dim=1)

        data.y_reg = ((radii[idx] - rg_mean) / rg_std).view(1)
        data.y_cls = (radii[idx] > median).float().view(1)
        data.model_id = idx + 1
        graphs.append(data)

    return graphs, rg_mean, rg_std


def make_model(torch, GCNConv, global_mean_pool, in_channels: int):
    class GraphRegressorClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, 64)
            self.reg_head = torch.nn.Linear(64, 1)
            self.cls_head = torch.nn.Linear(64, 1)

        def forward(self, batch):
            x = self.conv1(batch.x, batch.edge_index).relu()
            x = self.conv2(x, batch.edge_index).relu()
            pooled = global_mean_pool(x, batch.batch)
            return self.reg_head(pooled).squeeze(-1), self.cls_head(pooled).squeeze(-1)

    return GraphRegressorClassifier()


def main():
    torch, F, DataLoader, GCNConv, global_mean_pool = _require_pyg()
    torch.manual_seed(7)

    graphs, rg_mean, rg_std = build_dataset(torch)
    order = torch.randperm(len(graphs), generator=torch.Generator().manual_seed(7)).tolist()
    train_graphs = [graphs[i] for i in order[:14]]
    test_graphs = [graphs[i] for i in order[14:]]

    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=6)

    model = make_model(torch, GCNConv, global_mean_pool, graphs[0].x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(1, 81):
        model.train()
        total = 0.0
        for batch in train_loader:
            pred_reg, pred_cls = model(batch)
            loss_reg = F.mse_loss(pred_reg, batch.y_reg.view(-1))
            loss_cls = F.binary_cross_entropy_with_logits(pred_cls, batch.y_cls.view(-1))
            loss = loss_reg + loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * batch.num_graphs

        if epoch in {1, 20, 40, 80}:
            print(f"epoch {epoch:02d} train_loss={total / len(train_graphs):.3f}")

    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        pred_reg, pred_cls = model(batch)
        pred_rg = pred_reg * rg_std + rg_mean
        true_rg = batch.y_reg.view(-1) * rg_std + rg_mean
        mae = (pred_rg - true_rg).abs().mean()
        acc = ((pred_cls.sigmoid() > 0.5) == batch.y_cls.view(-1).bool()).float().mean()

    print("\nToy holdout metrics")
    print(f"  radius-of-gyration MAE: {mae:.3f} A")
    print(f"  expanded/constrained accuracy: {acc:.2%}")
    print("\nFirst holdout predictions")
    for model_id, predicted, observed in zip(batch.model_id.tolist(), pred_rg, true_rg):
        print(f"  model {model_id:02d}: predicted {predicted:.2f} A, observed {observed:.2f} A")


if __name__ == "__main__":
    main()
