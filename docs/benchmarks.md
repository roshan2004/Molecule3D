# Benchmarks

These benchmarks are small, reproducible checks for MolScope's core operations:
PDB parsing, residue contact maps, and graph export. They are meant to give a
practical sense of overhead on the bundled sample structures, not to replace
domain-specific benchmark suites for MDAnalysis, MDTraj, RDKit, or simulation
engines.

Run them locally from the repository root:

```bash
uv run python scripts/benchmark_core.py
```

## Local reference run

Measured on May 26, 2026 with Python 3.12.11 on macOS arm64. Each row reports
the median of 7 timed runs after one warm-up call.

| Structure | Operation | Median (ms) | Range (ms) | Notes |
| --- | --- | ---: | ---: | --- |
| `1fqy` | parse PDB | 3.37 | 3.27-4.09 | single 1,661-atom model |
| `1aml` | parse PDB models | 23.81 | 23.35-24.87 | 20-model NMR ensemble, 11,960 atoms total |
| `1fqy` CA | distance matrix (NumPy) | 1.25 | 1.13-1.54 | 226 x 226 dense CA matrix |
| `1fqy` | residue contact map | 1.49 | 1.49-1.55 | 226 x 226 residue map |
| `1aml` model 1 | residue contact map | 0.19 | 0.18-0.22 | 40 x 40 residue map |
| `1fqy` | graph export | 0.96 | 0.90-0.99 | inferred atom-level bonds |
| `1aml` model 1 | graph export | 0.43 | 0.40-1.22 | inferred atom-level bonds |

## What is measured

- `parse PDB`: fixed-column PDB parsing into `Molecule` objects.
- `parse PDB models`: all `MODEL` records from the NMR ensemble.
- `distance matrix`: `alpha_carbons().distance_matrix(backend="numpy")`.
- `residue contact map`: `contact_map(cutoff=8.0, level="residue")`.
- `graph export`: `to_graph()` using explicit or inferred bonds.

The benchmark script keeps parsed molecules in memory for contact-map and graph
export tests so those rows measure analysis/export cost rather than file I/O.
If PyTorch or CuPy is installed, the script also adds optional dense-backend
rows for CPU/GPU distance matrices.

## Interpreting the numbers

The bundled examples are intentionally small enough for teaching and CI. Larger
systems scale differently:

- PDB parsing scales approximately with line count.
- Residue-level contact maps are much smaller than atom-level dense maps.
- Atom-level dense contact outputs are still `O(N^2)` in memory.
- Bond/contact searches use SciPy KD-tree paths when SciPy is installed and
  fall back to pure NumPy paths otherwise.

Use the benchmark script as a smoke test when changing parsers, contact maps, or
graph export internals.
