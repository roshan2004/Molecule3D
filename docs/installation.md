# Installation

## From PyPI

```bash
pip install molscope
```

## From a local checkout

```bash
git clone https://github.com/roshan2004/molscope
cd molscope
uv sync
uv run pytest
```

## Optional dependencies

```bash
pip install "molscope[fast]"   # scipy KD-tree bond search
pip install "molscope[viz]"    # py3Dmol notebook viewer
pip install "molscope[graph]"  # NetworkX exporter only
```

PyTorch Geometric and DGL are optional manual installs because PyTorch builds
are platform-specific:

```bash
pip install torch torch_geometric
pip install dgl
```

## Documentation site

Build the docs locally with:

```bash
uv sync --group docs
uv run mkdocs serve
```
