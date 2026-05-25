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
    print(f"  {aqp.name}: {len(aqp)} atoms")
    print(f"  centre of mass:      {aqp.center_of_mass.round(2)}")
    print(f"  radius of gyration:  {aqp.radius_of_gyration:.2f} A")
    print(f"  inferred bonds:      {len(aqp.bonds())}")

    print("\n== 3. Chain transforms (each returns a new Molecule) ==")
    view = aqp.centered().rotate("z", 90).translate((1, 2, -1))
    print(f"  centroid after centre+rotate+translate: {view.centroid.round(3)}")

    print("\n== 4. Compare NMR ensemble models (1aml) ==")
    models = m3d.read_pdb_models("1aml.pdb")
    print(f"  {len(models)} models, {len(models[0])} atoms each")
    raw = models[0].rmsd(models[1])
    fit = models[0].rmsd(models[1], align=True)
    print(f"  RMSD model 1 vs 2:  {raw:.2f} A raw / {fit:.2f} A after Kabsch fit")

    print("\n== 5. Write a transformed structure back to disk ==")
    m3d.write_xyz(aqp.centered(), "aqp_centered.xyz")
    print("  wrote aqp_centered.xyz")

    print("\n== 6. Plot ==")
    show_or_save(aqp.centered(), "example_aquaporin.png")


if __name__ == "__main__":
    main()
