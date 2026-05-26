# Limitations

MolScope is intentionally lightweight. It is designed for teaching,
exploration, prototyping, and small ML workflows.

Known limitations:

- The built-in CIF reader handles standard `_atom_site` coordinate loops,
  quoted values, and semicolon text fields. Optional Gemmi-backed syntax,
  atom-site schema, and dictionary validation hooks are available through
  `[cif]`, with dictionary validation requiring local dictionary files.
- Bond inference is geometric and covalent-radius based. Explicit SDF bonds,
  SDF V2000 bond orders, and PDB `CONECT` records are preserved where present,
  and optional RDKit-backed chemical features are available via `[chem]`.
  MolScope still does not attempt general bond-order inference from raw
  coordinates.
- Coarse-graining creates beads and interpretable mapping reports, but it is not
  a full coarse-grained force-field engine.
- PyTorch Geometric and DGL have convenience extras (`[pyg]`, `[dgl]`, `[gnn]`),
  but specialized CUDA/ROCm/platform builds may still need backend-specific
  PyTorch installation steps.
- MolScope-native descriptors are practical fixed-size structural features.
  RDKit scalar descriptors are available through `[chem]`, but descriptor names
  and exact coverage follow the installed RDKit version.
- Full dense distance matrices and atom-level contact-map matrices are still
  O(N^2) outputs. Distance histograms, atom contact counts, and the no-SciPy
  `contacts()` fallback use chunked coordinate blocks, while SciPy enables a
  KD-tree contact search.
