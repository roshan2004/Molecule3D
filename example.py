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


def _save_contact_map(contact_map, filename):
    """Plot a contact map, saving to a PNG when running headless."""
    ax = contact_map.plot(show=not HEADLESS)
    if HEADLESS:
        ax.figure.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  saved contact map -> {filename}")


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

    print("\n== 6. Build a molecular graph (for ML) ==")
    graph = aqp.to_graph()
    print(f"  {graph.n_atoms} nodes, {graph.n_bonds} bonds")
    print(f"  node feature matrix: {graph.node_features().shape} [atomic_number, mass]")
    try:
        nxg = aqp.to_networkx()
        print(f"  networkx graph: {nxg.number_of_nodes()} nodes, "
              f"{nxg.number_of_edges()} edges")
    except ImportError:
        print("  (install networkx / torch_geometric / dgl to export to those)")

    print("\n== 7. Contact maps and ensemble variability ==")
    cmap = aqp.contact_map(cutoff=8.0, level="residue")
    print(f"  residue contact map: {cmap.matrix.shape}, "
          f"~{cmap.matrix.sum(1).mean():.1f} contacts/residue")
    freq = m3d.ensemble_contact_frequency(models, cutoff=8.0)
    variable = ((freq.matrix > 0) & (freq.matrix < 1)).sum() // 2
    print(f"  NMR ensemble: {variable} residue pairs in contact in only some models")

    print("\n== 8. Cluster the NMR ensemble by RMSD ==")
    clusters = m3d.cluster(models, n_clusters=3)
    print(f"  {len(models)} models -> {clusters.n_clusters} clusters, "
          f"sizes {[len(v) for v in clusters.groups().values()]}")
    print(f"  representative model per cluster: {clusters.representatives()}")

    print("\n== 9. Coarse-grain to one bead per residue ==")
    cg = aqp.coarse_grain("residue_com")
    print(f"  atomistic {len(aqp)} atoms -> CG {len(cg)} beads, {len(cg.bonds())} bonds")
    martini = aqp.coarse_grain("martini")
    print(f"  martini-like BB/SC model: {len(martini)} beads")

    print("\n== 10. Write a transformed structure back to disk ==")
    m3d.write_xyz(aqp.centered(), "aqp_centered.xyz")
    print("  wrote aqp_centered.xyz")

    print("\n== 11. Plot (colour by chain) and the contact map ==")
    show_or_save(aqp.centered(), "example_aquaporin.png", color_by="chain")
    _save_contact_map(cmap, "example_contactmap.png")
    print("\n== 12. Plot the coarse-grained bead network ==")
    show_or_save(cg, "example_cg.png", scale=200)


if __name__ == "__main__":
    main()
