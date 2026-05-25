# Limitations

MolScope is intentionally lightweight. It is designed for teaching,
exploration, prototyping, and small ML workflows.

Known limitations:

- The CIF reader is a basic parser for standard `_atom_site` coordinate loops,
  not a full mmCIF syntax implementation.
- Bond inference is geometric and covalent-radius based. It is not a chemistry
  toolkit with valence, aromaticity, charges, or bond-order perception.
- Coarse-graining creates beads and interpretable mapping reports, but it is not
  a full coarse-grained force-field engine.
- PyTorch Geometric and DGL exporters require manual backend installation.
- Descriptors are practical fixed-size features, not a complete cheminformatics
  descriptor library.
- Very large dense distance computations may require SciPy or custom batching.
