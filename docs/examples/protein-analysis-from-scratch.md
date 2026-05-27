# Protein analysis from scratch

This example treats proteins as structured coordinate data rather than just
points in 3D space. It uses all three bundled PDB structures:

- `1fqy.pdb`: backbone atoms, alpha carbons, contact maps and simplified
  secondary structure.
- `1aml.pdb`: NMR ensemble contact frequency.
- `3ptb.pdb`: ligands, waters and binding-site residues.

```python
import molscope as ms

aqp = ms.read("examples/data/1fqy.pdb")
print(aqp.summary())
print(len(aqp.backbone()), "backbone atoms")
print(len(aqp.alpha_carbons()), "alpha carbons")

cmap = aqp.contact_map(cutoff=8.0, level="residue", method="ca", min_seq_sep=4)
print(cmap.n_contacts, "non-local CA contacts")

ss = aqp.secondary_structure()
print(ss.summary())
```

For an NMR ensemble:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
print(len(models), "models")
print(freq.matrix.shape, freq.n_contacts)
```

For a protein-ligand complex:

```python
trypsin = ms.read("examples/data/3ptb.pdb")
print(trypsin.ligands())                 # [LigandResidue(A:BEN1, 9 atoms)]
print(len(trypsin.select(resname="HOH")), "water atoms")

site = trypsin.binding_site(cutoff=4.5)
for residue, distance in zip(site.residues[:8], site.min_distances[:8]):
    print(f"{residue!s:<10} {distance:.2f} A")
```

MolScope's `secondary_structure()` is a simplified, dependency-free DSSP-style
assignment for teaching and prototyping. It is not a bit-identical replacement
for canonical `mkdssp`.

Runnable versions:

```bash
uv run python examples/protein_analysis.py
uv run jupyter lab notebooks/protein_analysis_from_scratch.ipynb
```
