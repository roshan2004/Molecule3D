# Installation

## From PyPI

```bash
pip install molscope
```

The package is published on PyPI at
[pypi.org/project/molscope](https://pypi.org/project/molscope/).

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
pip install "molscope[graph]"  # NetworkX exporter
pip install "molscope[chem]"   # RDKit chemical perception
pip install "molscope[cif]"    # Gemmi CIF/mmCIF parser and validation helpers
pip install "molscope[gpu]"    # PyTorch dense distance/contact-map backend
pip install "molscope[pyg]"    # PyTorch + PyTorch Geometric exporter
pip install "molscope[dgl]"    # PyTorch + DGL exporter
pip install "molscope[gnn]"    # NetworkX + PyG + DGL exporters
```

The `gpu`, `pyg`, `dgl`, and `gnn` extras use the default PyPI PyTorch/DGL/PyG
packages. If you need a specific CUDA, ROCm, Apple Silicon, or cluster build,
install the matching PyTorch stack first from the backend project's
instructions, then install MolScope normally:

```bash
pip install torch torch_geometric
pip install dgl
pip install molscope
```

## Documentation site

Build the docs locally with:

```bash
uv sync --group docs
uv run mkdocs serve
```
