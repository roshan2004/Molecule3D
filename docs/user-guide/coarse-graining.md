# Coarse-Graining

Coarse-graining maps an atomistic structure onto beads:

```python
cg = mol.coarse_grain("residue_com")
cg = mol.coarse_grain("residue_centroid")
cg = mol.coarse_grain("martini")
```

The result is still a `Molecule`, so it can be plotted, transformed, converted
to a graph, and analyzed.

## Custom residue mappings

```python
mapping = {"ALA": {"BB": ["N", "CA", "C", "O"], "SC": ["CB"]}}
cg = mol.coarse_grain(mapping)
```

## Custom index mappings

```python
cg = mol.coarse_grain(
    {"head": [0, 1, 2, 3], "tail": [4, 5, 6, 7]},
    bonds=[("head", "tail")],
)
```

Name-based bonds are intended for unique bead names. Repeated names such as
`BB` and `SC` are ambiguous across residues; use bead indices for those.

## Mapping reports

```python
cg = mol.coarse_grain("martini")
print(cg.mapping_report())

cg, report = mol.coarse_grain(mapping, return_report=True)
```

MolScope is useful for interpretable coarse-graining prototypes and teaching.
It is not a complete force-field engine with bonded, nonbonded, angle, dihedral,
charge, exclusion, or topology export parameter handling.
