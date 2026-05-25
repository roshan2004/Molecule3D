# Coarse-Grain A Protein

```python
import molecule3d as m3d

mol = m3d.read("1fqy.pdb")

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
