# Protein analysis: secondary structure, interfaces, binding sites

Beyond contact maps, MolScope offers protein-specific analysis built on the
chain/residue metadata and the ATOM/HETATM flag preserved when reading PDB and
mmCIF files. None of it needs extra dependencies.

## Secondary structure elements

`secondary_structure()` runs the built-in DSSP and returns a
[`SecondaryStructure`][]; on top of the per-residue codes you can extract
elements, reduce to three states, and break the assignment down by chain.

```python
import molscope as ms

ss = ms.read("examples/data/1fqy.pdb").secondary_structure()

ss.string            # 8-state codes, e.g. '--HHHHH--EEEE--'
ss.simplified()      # 3-state: 'CCHHHHHCCEEEEC...' (H helix, E strand, C coil)

for seg in ss.segments():           # contiguous helices/strands, coil omitted
    print(seg)                      # e.g. SSSegment(H A:9-36, len=28)

ss.summary()                        # helix/strand/coil counts and fractions
ss.per_chain()                      # the same breakdown per chain id
```

## Backbone torsions (Ramachandran)

`backbone_torsions()` returns the phi/psi/omega dihedrals per residue in degrees,
with `NaN` at chain termini and breaks:

```python
tor = ms.read("examples/data/1fqy.pdb").backbone_torsions()
tor.phi, tor.psi, tor.omega         # (R,) arrays aligned with tor.resids/chains
```

Helical residues cluster near phi -63, psi -42; trans peptide bonds give omega
near 180. Plot `tor.phi` against `tor.psi` for a Ramachandran scatter.

## Chain interfaces

For multi-chain structures, find the residues that form an interface between two
chains, or summarise inter-chain contacts across the whole assembly:

```python
mol = ms.fetch("1brs")                       # barnase-barstar complex

iface = mol.interface("A", "D", cutoff=5.0)  # -> Interface
iface.residues_a                             # interface residues on chain A
iface.residues_b                             # ...and on chain D
iface.n_atom_contacts                        # atom pairs across the interface

ccm = mol.chain_contacts(cutoff=5.0)         # -> ChainContactMatrix
ccm.count("A", "D")                          # inter-chain atom contacts
```

## Ligand-binding sites

HETATM groups (ligands, cofactors, ions, water) are tracked separately from
polymer atoms. `ligands()` lists the non-solvent groups; `binding_site()` finds
the protein residues around one.

```python
mol = ms.read("examples/data/3ptb.pdb")      # trypsin + benzamidine

mol.ligands()                                # [LigandResidue(A:BEN1, 9 atoms)]

site = mol.binding_site(cutoff=4.5)          # the single ligand is auto-detected
for res, d in zip(site.residues, site.min_distances):
    print(res, round(d, 2))                  # closest residues first
# A:GLY219 2.82
# A:ASP189 2.87   <- the benzamidine specificity residue
# A:SER190 3.04
# ...
```

When several ligands are present, name one explicitly:

```python
mol.binding_site(ligand="BEN")               # by residue name
mol.binding_site(ligand=("A", 1))            # by (chain, resid)
```

Water and common ions are excluded from `ligands()` by default
(`exclude_water`, `exclude_ions`). The polymer/hetero split is also available as
selections:

```python
mol.protein()        # ATOM atoms only
mol.hetero_atoms()   # HETATM atoms only
mol.select(hetero=True)
```

[`SecondaryStructure`]: ../api-reference.md
