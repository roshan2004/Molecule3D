# Diverse selection from a molecule table

Most of MolScope works on a single 3D structure. The `molscope select` command
and the `molscope.library` module are the exception: they work on a *table of
molecules*, a CSV or XLSX with one row per compound and columns such as an id, a
SMILES string, and numeric properties.

The task they solve is library prep: given many candidates, pick `n` that are
spread out in descriptor space rather than clustered together.

## Quick start

Select on numeric columns that are already in the table (CSV needs no optional
dependency):

```bash
molscope select molecules.csv --descriptor-cols MW ALogP -n 100 --out picked.csv
```

Or compute descriptors from a SMILES column with RDKit, then select on those:

```bash
molscope select molecules.xlsx --smiles-col SMILES --compute-descriptors -n 100 --out picked.csv
```

The selected rows are written to `--out` (`.csv` or `.xlsx`); without `--out` the
command prints a one-line summary. When descriptors are computed, the new
descriptor columns are carried into the output table.

## How selection works

Selection is a **MaxMin** (farthest-first) traversal over the chosen descriptors:

1. Descriptors are z-scored by default so no single column dominates the distance
   purely because of its scale (pass `--no-standardize` to select on raw values).
2. The most extreme molecule (farthest from the centroid) is chosen first.
3. Each subsequent pick is the molecule that maximises the minimum distance to
   everything already chosen.

Selection is deterministic: the same table and `-n` always give the same subset.
Rows with any missing or non-numeric descriptor are excluded from the candidate
pool, and if fewer complete rows than `-n` exist, all of them are returned (with
a note on stderr).

## Descriptors from SMILES

With `--compute-descriptors --smiles-col COL`, MolScope computes a default set of
RDKit descriptors: `MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, and
`NumRotatableBonds`. `MolLogP` is RDKit's Crippen logP, the standard stand-in for
ALogP. Choose a different set with `--rdkit-descriptors NAME [NAME ...]` using any
[RDKit descriptor name](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).
Unparseable SMILES become rows with missing descriptors and are skipped.

## Python API

The same building blocks are available in `molscope.library`:

```python
from molscope.library import read_table, smiles_descriptors, select_diverse

table = read_table("molecules.csv")
matrix, names = smiles_descriptors(table.column("SMILES"))
table = table.with_columns(names, matrix)
picked = table.select_rows(select_diverse(matrix, 100))
picked.write("picked.csv")
```

## Installation

| Input / feature | Extra |
| --- | --- |
| CSV/TSV input and output, MaxMin selection | none (core) |
| `.xlsx` read/write | `pip install "molscope[xlsx]"` |
| `--compute-descriptors` from SMILES | `pip install "molscope[chem]"` |

## Scope

This is a lightweight, descriptor-space diversity pick for triage and prototyping.
It is not a replacement for dedicated chemical-diversity tooling: it does not do
fingerprint/Tanimoto MaxMin, clustering (Butina, sphere exclusion), or
property-weighted designs. For those, reach for RDKit's `SimDivFilters` or a
cheminformatics platform.
