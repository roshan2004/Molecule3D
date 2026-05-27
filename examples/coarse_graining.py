"""Educational coarse-graining workflow.

This example shows how MolScope maps atomistic coordinates to beads. It compares
one-bead-per-residue centre-of-mass mapping with a simplified backbone/sidechain
teaching model inspired by Martini concepts.

Run from the repository root:

    uv run python examples/coarse_graining.py

The script writes:

    docs/assets/coarsegrain/1fqy-cg-mapping-comparison.png

These mappings are for teaching, visual inspection and graph prototyping. They
are not production Martini topology or force-field generation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = ROOT / "docs" / "assets" / "coarsegrain" / "1fqy-cg-mapping-comparison.png"


def save_mapping_comparison(fragment, output: Path = OUT):
    """Save a two-panel atomistic-to-CG mapping comparison."""
    residue_com = fragment.coarse_grain("residue_com")
    bb_sc = fragment.coarse_grain("martini")

    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ms.plot_mapping(fragment, residue_com, ax=ax1, show=False)
    ax1.set_title("Residue COM mapping\none bead per residue")

    ms.plot_mapping(fragment, bb_sc, ax=ax2, show=False)
    ax2.set_title("Simplified BB/SC mapping\nbackbone and sidechain beads")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    mol = ms.read(DATA / "1fqy.pdb")
    fragment = mol.select(resid=(8, 12))

    residue_com = fragment.coarse_grain("residue_com")
    residue_centroid = fragment.coarse_grain("residue_centroid")
    bb_sc = fragment.coarse_grain("martini")

    first_shift = float(abs(residue_com.coords[0] - residue_centroid.coords[0]).max())
    print(f"Fragment: {len(fragment)} atoms across {len(residue_com)} residues")
    print(f"Residue COM: {len(residue_com)} beads, {len(residue_com.bonds())} bonds")
    print(f"Residue centroid: {len(residue_centroid)} beads")
    print(f"Max coordinate shift for first residue COM vs centroid: {first_shift:.3f} A")
    print(f"Simplified BB/SC: {len(bb_sc)} beads, {len(bb_sc.bonds())} bonds")
    print(bb_sc.coarse_grain_report.coverage())

    output = save_mapping_comparison(fragment)
    print(f"Saved {output}")
    print(
        "Limit: this is a bead assignment and simple bead graph for teaching; "
        "it is not a validated simulation topology or Martini parameter set."
    )


if __name__ == "__main__":
    main()
