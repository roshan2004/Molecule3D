"""Render documentation images for contact-map and distance-matrix examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import molscope as ms
from molscope.plotting import plot_distance_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = ROOT / "docs" / "assets" / "contactmaps"


def save(ax, path: Path) -> None:
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mol = ms.read(DATA / "1fqy.pdb")
    ca = mol.alpha_carbons()
    models = ms.read_pdb_models(DATA / "1aml.pdb")

    save(
        mol.contact_map(cutoff=8.0, level="residue").plot(show=False),
        OUT / "1fqy-residue-contact-map.png",
    )
    save(
        plot_distance_matrix(ca.distance_matrix(), show=False),
        OUT / "1fqy-ca-distance-matrix.png",
    )
    save(
        ms.ensemble_contact_frequency(models, cutoff=8.0).plot(show=False),
        OUT / "1aml-contact-frequency.png",
    )


if __name__ == "__main__":
    main()
