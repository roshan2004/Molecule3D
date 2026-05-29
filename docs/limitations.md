# Limitations by workflow

MolScope is intentionally lightweight. It is designed for teaching,
exploration, prototyping, and small ML workflows, not as a replacement for
specialist molecular simulation, cheminformatics, or visualisation systems.

This page lists the practical limits of each core workflow so you can decide
when MolScope is appropriate and when to reach for a specialist tool. For the
underlying scientific cross-checks and tolerances, see
[Scientific validation](validation.md).

## Scientific scope

MolScope is a structure-analysis and representation toolkit. It does not run
molecular dynamics, energy minimisation, docking, force-field assignment, or
production coarse-grained simulation setup.

Reach for specialist tools when you need validated production workflows:

- **RDKit** for full cheminformatics and robust chemistry perception.
- **MDAnalysis**, **MDTraj**, or similar for large trajectory analysis.
- **PyMOL**, **VMD**, **ChimeraX**, or NGL-based tools for deep interactive
  visualisation.
- **Martini tooling** for production coarse-grained model preparation.

## Inputs and file formats

These limits apply to every workflow, since each one starts from a parsed
`Molecule`.

The built-in readers cover common teaching and prototyping paths: XYZ
(including bare coordinate dumps), fixed-column PDB ATOM/HETATM records and
`CONECT`, multi-model NMR PDBs, standard mmCIF `_atom_site` coordinate loops,
and SDF/MOL V2000 atoms, bonds, bond orders and formal charges.

Important limits:

- The built-in CIF reader is **not** a complete mmCIF dictionary engine.
  Optional Gemmi-backed syntax, atom-site schema and dictionary validation are
  available through `pip install "molscope[cif]"`, but dictionary validation
  still needs local dictionary files.
- Alternate PDB conformations default to the primary conformation; use
  `read_pdb(..., altloc="first"|"highest_occupancy"|"all")` when needed.
- There is no trajectory format support (XTC, DCD, TRR, NetCDF). MolScope reads
  static structures and multi-model NMR files only.

## Descriptors workflow

`Molecule.descriptors()` and the optional RDKit-backed helpers produce
fixed-width feature tables for screening, QC and classical ML.

What it does:

- Preserves explicit SDF bonds, SDF V2000 bond orders, formal charges, and
  PDB `CONECT` records where present. When connectivity is missing, bonds are
  inferred geometrically from covalent radii.
- Returns deterministic, fixed-size structural descriptors suitable for
  side-by-side tables.

Limits:

- General bond-order inference from raw coordinates is **out of scope**. If
  your input lacks `CONECT`/SDF bond records, treat single-bond inference as a
  rough geometric guess rather than a chemistry-aware assignment.
- Aromaticity, valence, and RDKit scalar descriptors require
  `pip install "molscope[chem]"`. Without it the corresponding columns are
  missing rather than wrong.
- RDKit descriptor names and coverage track the installed RDKit version, so
  reproducibility across environments depends on pinning RDKit.
- MolScope-native descriptors are practical fixed-size structural features,
  not a replacement for curated QSAR libraries like Mordred or the full RDKit
  descriptor catalogue.

## Graph ML workflow

`MolecularGraph` and `ResidueContactGraph` build atom/bond or residue/contact
graphs suitable for message-passing experiments and PyTorch Geometric
prototyping.

What it does:

- Exposes deterministic node/edge feature presets (`default`, `basic`, `ml`),
  Laplacian and random-walk positional encodings, and exporters to NetworkX,
  PyTorch Geometric, and DGL.
- Falls back to geometric bond inference when input connectivity is missing,
  so a graph is produced regardless of input format.

Limits:

- Node features are pragmatic basics: atomic number, mass, formal charge,
  aromatic flag, and a fixed element one-hot. There is **no** geometric basis
  (SchNet-style radial filters, DimeNet angular features, equivariant
  representations). Use a dedicated featuriser like DGL-LifeSci, OGB, or
  e3nn-based stacks for state-of-the-art GNN inputs.
- Default edge features are distance only. `basic` and `ml` add bond order and
  one-hot bond type where chemical perception is available; bond angles and
  dihedrals are not edge attributes.
- Residue contact edges come from a distance cutoff on chosen representative
  atoms. They do not encode hydrogen bonds, salt bridges, or any
  chemistry-aware contact taxonomy.
- When `CONECT`/SDF bonds are absent, geometrically-inferred bonds **flow
  through into the edge set** — be explicit about your input source if you
  care about edge fidelity.
- Laplacian and random-walk PEs are computed eagerly with dense NumPy
  (`(N, N)` adjacency, `eigh` on the normalised Laplacian). They are fine for
  the small-to-medium graphs MolScope targets but are not suitable for very
  large systems.
- PyTorch Geometric export requires `pip install "molscope[pyg]"`. Tensors are
  built on CPU; MolScope provides no batched `DataLoader`, no transforms, and
  no training utilities.
- DGL export requires `dgl`, which is harder to install than the other
  backends: recent DGL releases ship **only `win_amd64` wheels on PyPI**, so
  on Linux and macOS you need DGL's own wheel index
  (`https://data.dgl.ai/wheels/...`). CI does not exercise the DGL exporter
  for this reason; treat it as best-effort.

## DSSP (secondary structure) workflow

`Molecule.secondary_structure()` implements a simplified, dependency-free DSSP
style assignment based on backbone hydrogen-bond patterns.

What is validated:

- The validation suite compares MolScope's simplified DSSP against `mkdssp`
  across three fold classes where the reference binary is available:
  Aquaporin-1 (`1fqy`, helix-dominated), ubiquitin (`1ubq`, mixed alpha/beta),
  and the SH3 domain (`1shg`, all-beta).
- After reducing DSSP states to helix/strand/coil, agreement is high across
  all three folds: about 99% on the helical and mixed structures and about
  98% on the all-beta one.

Limits:

- Not bit-identical to reference `mkdssp`. Treat output as the
  educational/prototyping equivalent of DSSP, not as a substitute for it in
  production pipelines.
- Disagreements concentrate at the boundary residues of helices and strands,
  and on irregular or low-quality structures, rather than on any one fold
  class.
- Needs backbone N/CA/C/O atoms, so bare XYZ input is insufficient.
- Only the standard collapsed states (H/E/C) are validated against the
  reference; finer DSSP categories are not.

## Coarse-grained beads workflow

`coarse_grain()` maps atomistic structures to residue centroids, residue
centres of mass, simplified Martini-style backbone/side-chain beads, or
custom bead mappings, and reports assignment statistics.

What it does:

- Produces bead coordinates and a `Molecule` so plotting and graph export
  still work.
- Builds simple bead connectivity.
- Reports assigned and dropped atoms.
- Preserves explicitly requested virtual sites as derived coordinate metadata.

What it does **not** do:

- It is **not** a validated Martini force-field generator.
- It does not assign production force-field parameters.
- It does not build simulation-ready topologies.
- It does not write GROMACS `[ virtual_sites* ]` topology sections.
- It does not validate bead chemistry, elastic networks, or force constants.
- It does not claim thermodynamic, kinetic, or structural fidelity for a CG
  model without external validation.

Use these tools for teaching, mapping inspection, and graph prototyping before
moving to a production coarse-grained workflow.

## GPU and contact maps

Contact maps, distance matrices and the optional GPU backend live in the same
performance corner of the library, so their limits are related.

- Full dense distance matrices and atom-level contact-map matrices are
  `O(N^2)` outputs, even when routed through the optional Torch GPU backend.
  GPU makes the maths faster; it does not change the memory cost.
- Distance histograms, atom contact counts, and the no-SciPy `contacts()`
  fallback paths use chunked coordinate blocks, so they scale to larger
  systems than the dense outputs do.
- SciPy enables KD-tree bond/contact searches; install with
  `pip install "molscope[fast]"` for large structures.
- The Torch GPU path requires `pip install "molscope[gpu]"` and a working
  CUDA / MPS PyTorch install. There is no CuPy backend.
- MolScope is **not** a trajectory engine. Use MDAnalysis or MDTraj for any
  workload that means iterating contacts or distances over many frames.

See [Benchmarks](benchmarks.md) for a small reproducible timing page.

## See also

- [Scientific validation](validation.md) — invariants, reference-tool
  comparisons, assumptions, and tolerances.
- [Benchmarks](benchmarks.md) — reproducible timings on the bundled inputs.
