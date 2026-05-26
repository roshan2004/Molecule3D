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
D_gpu = mol.distance_matrix(backend="torch", device="cuda")  # optional
pairs = mol.contacts(cutoff=5.0)
count = mol.contact_count(cutoff=5.0)
```

`distance_matrix()` returns the full dense `N x N` matrix. It supports NumPy,
PyTorch, CuPy, and `auto` dense backends. `contacts()` uses a KD-tree when SciPy
is installed and a chunked fallback otherwise; tune the fallback with
`chunk_size=`. See [Contact maps and distance matrices](contact-maps.md) for
backend details, images, and benchmarks.

Rigid-body alignment:

```python
rmsd = a.rmsd(b, align=True)
aligned = a.superpose(b)
```

Bond inference uses covalent radii. If SciPy is installed, MolScope uses a
KD-tree path for larger structures.
