# Geometry And Measurements

Basic molecular properties:

```python
mol.centroid
mol.center_of_mass
mol.radius_of_gyration
mol.dimensions
mol.formula
```

Distances and angles:

```python
mol.distance(i, j)
mol.angle(i, j, k)
mol.dihedral(a, b, c, d)
```

Pairwise distances and contacts:

```python
D = mol.distance_matrix()
pairs = mol.contacts(cutoff=5.0)
```

Rigid-body alignment:

```python
rmsd = a.rmsd(b, align=True)
aligned = a.superpose(b)
```

Bond inference uses covalent radii. If SciPy is installed, Molecule3D uses a
KD-tree path for larger structures.
