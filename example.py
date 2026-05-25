"""Worked example of the molecule3d API on the bundled sample structures.

Run it from the repo root:

    uv run python example.py          # opens plot windows
    MPLBACKEND=Agg uv run python example.py   # headless: saves PNGs instead

It reads each sample file, prints a few derived properties, compares the NMR
models of 1aml, writes a transformed structure back out, and renders a plot.
"""

import os

import matplotlib

import molecule3d as m3d
from molecule3d import ensemble

# Use a non-interactive backend (save to file) when there's no display.
HEADLESS = matplotlib.get_backend().lower() == "agg" or not os.environ.get("DISPLAY")
if os.environ.get("MPLBACKEND", "").lower() == "agg":
    HEADLESS = True


def show_or_save(molecule, filename, **kwargs):
    """Plot a molecule, saving to a PNG when running headless."""
    ax = molecule.plot(show=not HEADLESS, **kwargs)
    if HEADLESS:
        ax.figure.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  saved plot -> {filename}")


def main():
    print("== 1. Read and inspect an .xyz file ==")
    helix = m3d.read("helix_201.xyz")
    print(f"  {helix.name}: {len(helix)} atoms")
    print(f"  centroid: {helix.centroid.round(3)}")

    print("\n== 2. Read a protein (.pdb) and derive properties ==")
    aqp = m3d.read("1fqy.pdb")  # Aquaporin-1
    print(f"  {aqp.summary()}")
    print(f"  centre of mass:      {aqp.center_of_mass.round(2)}")
    print(f"  radius of gyration:  {aqp.radius_of_gyration:.2f} A")
    print(f"  inferred bonds:      {len(aqp.bonds())}")

    print("\n== 3. Select atoms by metadata ==")
    ca = aqp.alpha_carbons()
    print(f"  alpha-carbons: {len(ca)} of {len(aqp)} atoms")
    print(f"  N-CA-C angle of residue 1: {aqp.angle(0, 1, 2):.1f} deg")

    print("\n== 4. Chain transforms (each returns a new Molecule) ==")
    view = aqp.centered().rotate("z", 90).translate((1, 2, -1))
    print(f"  centroid after centre+rotate+translate: {view.centroid.round(3)}")

    print("\n== 5. Compare NMR ensemble models (1aml), CA-based ==")
    models = m3d.read_pdb_models("1aml.pdb")
    ca_models = [m.alpha_carbons() for m in models]
    print(f"  {len(models)} models, {len(ca_models[0])} CA atoms each")
    raw = ca_models[0].rmsd(ca_models[1])
    fit = ca_models[0].rmsd(ca_models[1], align=True)
    print(f"  CA-RMSD model 1 vs 2:  {raw:.2f} A raw / {fit:.2f} A after Kabsch fit")
    fluct = ensemble.rmsf(ca_models)
    print(f"  most flexible residue: #{fluct.argmax() + 1} (RMSF {fluct.max():.2f} A)")

    print("\n== 6. Write a transformed structure back to disk ==")
    m3d.write_xyz(aqp.centered(), "aqp_centered.xyz")
    print("  wrote aqp_centered.xyz")

    print("\n== 7. Plot (colour by chain) ==")
    show_or_save(aqp.centered(), "example_aquaporin.png", color_by="chain")


if __name__ == "__main__":
    main()
