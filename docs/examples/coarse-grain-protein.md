# Coarse-Grain A Protein

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")

cg = mol.coarse_grain("residue_com")
print(cg.summary())
print(cg.mapping_report())

cg.plot(scale=200)
```

For a Martini-like teaching model:

```python
cg = mol.coarse_grain("martini")
G = cg.to_graph()
```

## See the mapping

`plot_mapping` overlays the beads on the atoms they replace, colouring each atom
by its bead. A short fragment reads most clearly:

```python
fragment = mol.select(resid=(8, 12))
ms.plot_mapping(fragment, fragment.coarse_grain("martini"))
```

## Inspect and export the assignment

```python
report = cg.coarse_grain_report
print(report.coverage())              # beads / atoms covered
print(report.beads[0].atom_indices)    # which atoms became bead 0

cg.write_mapping("mapping.json")      # JSON record (round-trippable)
cg.write_index("mapping.ndx")         # GROMACS-style index, one group per bead

record = ms.read_cg_mapping("mapping.json")
cg_again = ms.apply_cg_mapping(mol, record)   # rebuild on the same structure
```
