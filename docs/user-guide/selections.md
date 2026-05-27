# Selections

PDB files, and standard mmCIF atom-site loops, carry atom and residue metadata.
Use that metadata to create subsets:

```python
chain_a = mol.select(chain="A")
carbons = mol.select(element="C")
waters = mol.select(resname="HOH")
region = mol.select(resid=(10, 20))
```

Protein helpers:

```python
backbone = mol.backbone()
ca = mol.alpha_carbons()
```

Numpy-style masks and index arrays also work:

```python
first_ten = mol[list(range(10))]
```

Selections return new `Molecule` objects.

The command-line viewer accepts the same basic fields for quick inspection:

```bash
molscope examples/data/1fqy.pdb --select "chain=A and atom_name=CA"
molscope examples/data/1fqy.pdb --select chain=A --select atom_name=CA
```
