# Limitations and validation

MolScope is intentionally lightweight. It is designed for teaching,
exploration, prototyping, and small ML workflows, not as a replacement for
specialist molecular simulation, cheminformatics, or visualisation systems.

## Scientific scope

MolScope is a structure-analysis and representation toolkit. It does not run
molecular dynamics, energy minimisation, docking, force-field assignment, or
production coarse-grained simulation setup.

Use specialist tools when you need validated production workflows:

- RDKit for full cheminformatics and robust chemistry perception.
- MDAnalysis, MDTraj, or similar packages for large trajectory analysis.
- PyMOL, VMD, ChimeraX, or NGL-based tools for deep interactive visualisation.
- Martini tooling for production coarse-grained model preparation.

## File formats

The built-in readers cover common teaching and prototyping paths:

- XYZ, including bare coordinate dumps.
- Fixed-column PDB ATOM/HETATM records, PDB `CONECT`, and multi-model NMR files.
- Standard mmCIF `_atom_site` coordinate loops.
- SDF/MOL V2000 atoms, bonds, bond orders, and formal charges.

Important limits:

- The built-in CIF reader is not a complete mmCIF dictionary engine.
- Optional Gemmi-backed syntax, atom-site schema, and dictionary validation are
  available through `pip install "molscope[cif]"`, but dictionary validation
  requires local dictionary files.
- Alternate PDB conformations default to the primary conformation; use
  `read_pdb(..., altloc="first"|"highest_occupancy"|"all")` when needed.

## Bond perception and descriptors

MolScope preserves explicit SDF bonds, SDF V2000 bond orders, formal charges,
and PDB `CONECT` records where present. When connectivity is missing, bonds are
inferred geometrically from covalent radii.

Limitations:

- General bond-order inference from raw coordinates is out of scope.
- Aromaticity, valence, and RDKit scalar descriptors require
  `pip install "molscope[chem]"`.
- RDKit descriptor names and coverage follow the installed RDKit version.
- MolScope-native descriptors are practical fixed-size structural features, not
  a replacement for curated QSAR descriptor libraries.

## Secondary structure

`Molecule.secondary_structure()` implements a simplified, dependency-free DSSP
style assignment based on backbone hydrogen-bond patterns.

What is validated:

- The validation suite compares MolScope's simplified DSSP against `mkdssp`
  where the reference binary is available.
- On the bundled Aquaporin-1 structure (`1fqy`), CI records about 99% agreement
  after reducing DSSP states to helix/strand/coil.

Limitations:

- It is not bit-identical to reference `mkdssp`.
- Strand-rich and edge-case folds can disagree more strongly than helical test
  structures.
- It needs backbone N/CA/C/O atoms, so bare XYZ input is insufficient.
- Treat it as educational/prototyping secondary structure, not as a replacement
  for a reference DSSP installation in production pipelines.

## Coarse-graining

MolScope can map atomistic structures to residue centroids, residue centres of
mass, simplified Martini-style backbone/side-chain beads, or custom bead
mappings.

What it does:

- Creates bead coordinates.
- Builds simple bead connectivity.
- Reports assigned and dropped atoms.
- Returns a normal `Molecule` so plotting and graph export still work.

What it does not do:

- It is not a validated Martini force-field generator.
- It does not assign production force-field parameters.
- It does not build simulation-ready topologies.
- It does not validate bead chemistry, elastic networks, or force constants.

Use these tools for teaching, mapping inspection, and graph prototyping before
moving to a production coarse-grained workflow.

## Performance and scaling

MolScope is optimized for small-to-medium static structures and examples that
fit comfortably in memory.

- PDB parsing scales approximately with line count.
- Full dense distance matrices and atom-level contact-map matrices are `O(N^2)`
  outputs.
- Distance histograms, atom contact counts, and no-SciPy `contacts()` fallback
  paths use chunked coordinate blocks.
- SciPy enables KD-tree bond/contact searches.
- MolScope is not a trajectory engine; use MDAnalysis or MDTraj for large
  trajectory workloads.

See [Benchmarks](benchmarks.md) for a small reproducible timing page.

## Validation suite

Run validation locally with:

```bash
uv run pytest tests/validation -v -rs -s
```

The suite has two layers:

- Dependency-free invariants that should hold everywhere, such as rigid-body
  geometry behavior and coarse-grain conservation checks.
- Reference-tool comparisons that run when optional scientific tools are
  installed, including MDAnalysis, RDKit, and `mkdssp`.

CI runs the validation job separately from the main test matrix so failures in
scientific cross-checks are visible without slowing every basic test run.
