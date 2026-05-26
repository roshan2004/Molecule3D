"""Render documentation images for the coarse-graining workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = ROOT / "docs" / "assets" / "coarsegrain"


def save(ax, path: Path) -> None:
    ax.figure.set_size_inches(7, 6)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mol = ms.read(DATA / "1fqy.pdb")

    # A short, legible slice shows the atom->bead assignment (and bead legend)
    # clearly without crowding the figure.
    fragment = mol.select(resid=(8, 12))
    cg = fragment.coarse_grain("martini")
    save(
        ms.plot_mapping(fragment, cg, show=False),
        OUT / "1fqy-martini-mapping.png",
    )


if __name__ == "__main__":
    main()
