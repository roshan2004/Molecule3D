# Ligand binding site

Detect a bound ligand and report the protein residues that surround it, using
the bundled trypsin-benzamidine complex (3PTB).

```python
import molscope as ms

mol = ms.read("examples/data/3ptb.pdb")

# Water and ions are filtered out; only the real ligand remains.
print(mol.ligands())                 # [LigandResidue(A:BEN1, 9 atoms)]

site = mol.binding_site(cutoff=4.5)  # single ligand auto-detected
print(site)                          # BindingSite(A:BEN1: 13 residues < 4.5 A)

for res, dist in zip(site.residues, site.min_distances):
    print(f"{res!s:<10} {dist:.2f} A")
# A:GLY219   2.82
# A:ASP189   2.87   <- benzamidine specificity residue
# A:SER190   3.04
# A:GLY226   3.37
# A:SER195   3.65   <- catalytic serine
```

For quick figures or reports, convert the site to table-friendly residue
records and extract descriptors for only the site residues:

```python
site.to_records()[0]
# {'residue_id': 'A:GLY219', 'chain': 'A', 'resid': 219,
#  'insertion_code': '', 'resname': 'GLY',
#  'min_distance': 2.815..., 'n_atom_contacts': 5}

site.descriptors(mol, preset="pocket-basic")
site.plot(mol, show=False)          # pocket residues plus ligand
```

The same residue table is available from the command line:

```bash
molscope binding-site examples/data/3ptb.pdb --out site.csv --cutoff 4.5
```

Add `--descriptors-out pocket.csv` to also write the one-row
`pocket-basic` descriptor table.

When a structure has several ligands, select one by residue name or location:

```python
mol.binding_site(ligand="BEN")
mol.binding_site(ligand=("A", 1))
mol.binding_site(ligand=("A", 100, "A"))  # with insertion code
```

The CLI accepts the same choices as `--ligand BEN`, `--ligand A:1`, or
`--ligand A:100:A`.

A runnable version lives in
[`examples/binding_site.py`](https://github.com/roshan2004/molscope/blob/main/examples/binding_site.py).
See the full guide:
[Protein analysis](../user-guide/protein-analysis.md).
