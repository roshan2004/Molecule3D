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

    # 3. Secondary-structure overview of the protein.
    ss = mol.secondary_structure()
    summary = ss.summary()
    print(
        f"\nSecondary structure: {summary['helix']} helix, "
        f"{summary['strand']} strand, {summary['coil']} coil residues"
    )
    print(f"  elements: {len(ss.segments())} (helices/strands)")


if __name__ == "__main__":
    main()
