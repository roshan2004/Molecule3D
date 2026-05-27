"""A tour of MolScope's molecular-geometry tools.

Prints each geometric quantity on the bundled structures and (headless-safe)
renders the documentation figures. Pure NumPy + matplotlib.

    uv run python examples/geometry.py
    MPLBACKEND=Agg uv run python examples/geometry.py   # no display
"""

from pathlib import Path

import numpy as np

import molscope as ms

DATA = Path(__file__).resolve().parent / "data"


def main():
    mol = ms.read(str(DATA / "1fqy.pdb"))
    print(f"Loaded {mol.summary()}\n")

    # 1. Distances, angles, dihedrals (pick four atoms by index).
    i, j, k, m = 0, 10, 20, 30
    print("Local geometry")
    print(f"  distance({i},{j})        = {mol.distance(i, j):.3f} Å")
    print(f"  angle({i},{j},{k})       = {mol.angle(i, j, k):.2f}°")
    print(f"  dihedral({i},{j},{k},{m}) = {mol.dihedral(i, j, k, m):.2f}°")

    # 2. Centroid vs centre of mass: unweighted vs mass-weighted centre.
    print("\nCentres")
    print(f"  centroid       = {np.round(mol.centroid, 2)}")
    print(f"  center_of_mass = {np.round(mol.center_of_mass, 2)}")
    print(f"  offset between them = {np.linalg.norm(mol.centroid - mol.center_of_mass):.3f} Å")

    # 3. Size and shape: radius of gyration and the inertia tensor.
    print("\nSize and shape")
    print(f"  radius_of_gyration = {mol.radius_of_gyration:.2f} Å")
    print(f"  principal_moments  = {np.round(mol.principal_moments(), 1)}")
    print("  principal_axes (columns):")
    for row in np.round(mol.principal_axes(), 3):
        print(f"    {row}")

    # 4. Kabsch alignment and RMSD between two NMR models.
    models = ms.read_pdb_models(str(DATA / "1aml.pdb"))
    rmsd_raw = models[0].rmsd(models[1])
    rmsd_fit = models[0].rmsd(models[1], align=True)
    print("\nAlignment (1aml models 1 vs 2)")
    print(f"  RMSD as-is          = {rmsd_raw:.2f} Å")
    print(f"  RMSD after Kabsch   = {rmsd_fit:.2f} Å")

    # 5. Per-atom RMSF across the whole ensemble.
    rmsf = ms.ensemble.rmsf(models)
    print("\nFlexibility across the ensemble")
    print(f"  mean Cα-equivalent RMSF = {rmsf.mean():.2f} Å (max {rmsf.max():.2f} Å)")


if __name__ == "__main__":
    main()
