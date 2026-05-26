# PDB to PyTorch Geometric ML toy model

This tutorial shows the full path from a PDB file to graph ML:

1. read a multi-model PDB structure,
2. convert each model to a PyTorch Geometric graph,
3. train a small graph neural network with one regression head and one
   classifier head.

The dataset is intentionally tiny: the bundled `examples/data/1aml.pdb` NMR
ensemble has 20 conformers. The goal is to show the workflow clearly, not to
claim a scientifically meaningful predictor.

## Install the optional ML stack

Install PyTorch and PyTorch Geometric for your platform first:

```bash
uv pip install torch torch_geometric
.venv/bin/python examples/pdb_to_pyg_ml.py
```

Use `.venv/bin/python` directly in this repo because `uv run` may re-sync the
locked environment and remove optional packages that are not core MolScope
dependencies.

## Build graph labels from a PDB ensemble

Each NMR model becomes one graph. The regression target is radius of gyration.
The classification target is whether the model is more expanded than the median
model in the ensemble.

```python
from pathlib import Path

import torch
import molscope as ms

models = ms.read_pdb_models(Path("examples/data/1aml.pdb"))
radii = torch.tensor([m.radius_of_gyration for m in models], dtype=torch.float)
median = radii.median()

graphs = []
for idx, mol in enumerate(models):
    data = mol.to_pyg_data(node_preset="ml", edge_preset="ml")
    data.y_reg = radii[idx].view(1)
    data.y_cls = (radii[idx] > median).float().view(1)
    graphs.append(data)
```

`to_pyg_data()` populates PyTorch Geometric fields such as `x`, `edge_index`,
`edge_attr`, `pos`, and `z`. For this toy geometry task, append centered
coordinates to the node features so a simple GCN can see conformational shape:

```python
for data in graphs:
    centered_pos = data.pos - data.pos.mean(dim=0, keepdim=True)
    data.x = torch.cat([data.x, centered_pos], dim=1)
```

## Train a classifier/regressor

The example script uses a compact two-layer GCN with two graph-level heads:

```python
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphRegressorClassifier(torch.nn.Module):
    def __init__(self, in_channels):
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
```

The loss combines mean-squared error for the radius-of-gyration regression task
and binary cross-entropy for the expanded/constrained classification task.

```python
pred_reg, pred_cls = model(batch)
loss_reg = torch.nn.functional.mse_loss(pred_reg, batch.y_reg.view(-1))
loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
    pred_cls,
    batch.y_cls.view(-1),
)
loss = loss_reg + loss_cls
```

## Run the complete script

The runnable version lives at `examples/pdb_to_pyg_ml.py`:

```bash
.venv/bin/python examples/pdb_to_pyg_ml.py
```

Typical output prints training loss, a toy holdout radius-of-gyration MAE, and
expanded/constrained classification accuracy. The exact numbers are not the
point; the important part is the pipeline:

```text
PDB ensemble -> MolScope Molecule objects -> PyG Data graphs -> graph-level ML
```

For real work, replace the toy labels with experimental values, simulation
outputs, docking scores, functional classes, or other graph-level targets.
