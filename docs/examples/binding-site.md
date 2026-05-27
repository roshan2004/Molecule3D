# Ligand binding site

Detect a bound ligand and report the protein residues that surround it, using
the bundled trypsin-benzamidine complex (3PTB).

```python
import molscope as ms

mol = ms.read("examples/data/3ptb.pdb")

# Water and ions are filtered out; only the real ligand remains.
print(mol.ligands())                 # [LigandResidue(A:BEN1, 9 atoms)]

site = mol.binding_site(cutoff=4.5)  # single ligand auto-detected
print(site)                          # BindingSite(BEN1: 13 residues < 4.5 A)

for res, dist in zip(site.residues, site.min_distances):
    print(f"{res!s:<10} {dist:.2f} A")
# A:GLY219   2.82
# A:ASP189   2.87   <- benzamidine specificity residue
# A:SER190   3.04
# A:GLY226   3.37
# A:SER195   3.65   <- catalytic serine
```

When a structure has several ligands, select one by residue name or location:

```python
mol.binding_site(ligand="BEN")
mol.binding_site(ligand=("A", 1))
```

A runnable version lives in
[`examples/binding_site.py`](https://github.com/roshan2004/molscope/blob/main/examples/binding_site.py).
See the full guide:
[Protein analysis](../user-guide/protein-analysis.md).
