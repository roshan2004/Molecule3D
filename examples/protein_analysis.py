"""Protein-structure analysis from raw PDB coordinates.

Uses the three bundled protein examples:

* 1FQY: aquaporin-1, useful for backbone atoms, alpha carbons, contact maps and
  helix-rich secondary structure.
* 1AML: a 20-model NMR ensemble, useful for contact-frequency analysis.
* 3PTB: trypsin + benzamidine, useful for waters, ligands and binding sites.

Run from the repository root:

    uv run python examples/protein_analysis.py
"""

from pathlib import Path

import molscope as ms

DATA = Path(__file__).resolve().parent / "data"


def describe_structure(path: Path):
    mol = ms.read(path)
    residues = list(mol.residue_groups())
    backbone = mol.backbone()
    alpha_carbons = mol.alpha_carbons()
    water_atoms = len(mol.select(resname="HOH")) if "HOH" in set(mol.resnames) else 0

    print(f"{path.name}: {mol.summary()}")
    print(
        f"  residues={len(residues)} chains={mol.chain_ids()} "
        f"backbone_atoms={len(backbone)} alpha_carbons={len(alpha_carbons)}"
    )
    print(
        f"  protein_atoms={len(mol.protein())} hetero_atoms={len(mol.hetero_atoms())} "
        f"waters={water_atoms} ligands={mol.ligands()}"
    )
    return mol


def aquaporin_contacts_and_ss(mol):
    print("\n1FQY contact map and simplified DSSP")
    cmap = mol.contact_map(cutoff=8.0, level="residue", method="ca", min_seq_sep=4)
    print(f"  non-local CA contacts: {cmap.n_contacts}")
    print(f"  relative contact order: {cmap.contact_order():.3f}")

    ss = mol.secondary_structure()
    summary = ss.summary()
    print(
        "  simplified DSSP summary: "
        f"{summary['helix']} helix, {summary['strand']} strand, {summary['coil']} coil"
    )
    print(f"  first secondary-structure elements: {ss.segments()[:3]}")


def nmr_contact_frequency():
    print("\n1AML NMR ensemble contact frequency")
    models = ms.read_pdb_models(DATA / "1aml.pdb")
    freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
    radii = [model.radius_of_gyration for model in models]
    print(f"  models: {len(models)}")
    print(f"  contact-frequency matrix: {freq.matrix.shape}")
    print(f"  pairs observed at least once: {freq.n_contacts}")
    print(f"  radius of gyration range: {min(radii):.2f}-{max(radii):.2f} A")


def trypsin_binding_site(mol):
    print("\n3PTB ligand and binding site")
    site = mol.binding_site(cutoff=4.5)
    print(f"  ligand: {site.ligand}")
    print(f"  binding-site residues: {len(site.residues)}")
    for residue, distance in zip(site.residues[:8], site.min_distances[:8]):
        print(f"    {residue!s:<10} {distance:.2f} A")


def main():
    aquaporin = describe_structure(DATA / "1fqy.pdb")
    describe_structure(DATA / "1aml.pdb")
    trypsin = describe_structure(DATA / "3ptb.pdb")

    aquaporin_contacts_and_ss(aquaporin)
    nmr_contact_frequency()
    trypsin_binding_site(trypsin)

    print(
        "\nNote: MolScope secondary_structure() is a simplified educational DSSP-style "
        "assignment, not a bit-identical replacement for reference mkdssp."
    )


if __name__ == "__main__":
    main()
