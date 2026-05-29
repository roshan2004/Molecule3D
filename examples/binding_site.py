"""Ligand-binding-site and secondary-structure analysis over a real complex.

Reads the bundled trypsin-benzamidine structure (3PTB), detects the bound
ligand, reports the binding-site residues by distance, and prints a secondary
structure summary. Pure NumPy + matplotlib; no optional dependencies.

    uv run python examples/binding_site.py
"""

from pathlib import Path

import molscope as ms

DATA = Path(__file__).resolve().parent / "data"
STRUCTURE = DATA / "3ptb.pdb"


def main():
    mol = ms.read(str(STRUCTURE))
    print(f"Loaded {mol.summary()}\n")
    print(f"polymer atoms: {len(mol.protein())} | hetero atoms: {len(mol.hetero_atoms())}")

    # 1. Ligand detection (water and ions are filtered out).
    ligs = mol.ligands()
    print(f"\nDetected ligand(s): {ligs}")

    # 2. Binding-site residues around the ligand, closest first.
    site = mol.binding_site(cutoff=4.5)
    print(f"\n{site}")
    for res, dist in zip(site.residues, site.min_distances):
        print(f"  {res!s:<10} {dist:.2f} A")

    # 3. A table-friendly view for quick figures or reports.
    print("\nPer-residue contact table:")
    for row in site.to_records():
        label = row["residue_id"]
        print(
            f"  {label:<10} {row['min_distance']:.2f} A  "
            f"{row['n_atom_contacts']:>3} atom contacts"
        )

    # 4. Pocket descriptors for the binding-site residues and ligand contacts.
    site_desc = site.descriptors(mol, preset="pocket-basic")
    print(
        "\nBinding-site descriptors: "
        f"{int(site_desc['pocket_n_atoms'])} atoms, "
        f"{int(site_desc['pocket_n_residues'])} residues, "
        f"Rg {site_desc['pocket_radius_of_gyration']:.2f} A, "
        f"{int(site_desc['pocket_atom_contact_count'])} atom contacts"
    )

    # 5. Secondary-structure overview of the protein.
    ss = mol.secondary_structure()
    summary = ss.summary()
    print(
        f"\nSecondary structure: {summary['helix']} helix, "
        f"{summary['strand']} strand, {summary['coil']} coil residues"
    )
    print(f"  elements: {len(ss.segments())} (helices/strands)")


if __name__ == "__main__":
    main()
