"""Render documentation images for the molecular-geometry guide.

Produces two figures under docs/assets/geometry/:
  - principal axes of inertia and the centre of mass drawn on 1fqy, with Rg;
  - a per-residue RMSF profile across the 1aml NMR ensemble.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = ROOT / "docs" / "assets" / "geometry"


def render_principal_axes(path: Path) -> None:
    """Scatter the structure, mark the COM, and draw the three principal axes."""
    mol = ms.read(DATA / "1fqy.pdb")
    com = mol.center_of_mass
    axes = mol.principal_axes()                 # columns, ascending moment
    coords = mol.coords - com
    extent = np.abs(coords).max()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2, alpha=0.12, color="#888")

    colors = ["#e6194b", "#3cb44b", "#4363d8"]
    for k in range(3):
        v = axes[:, k] * extent * 0.9
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[k], linewidth=2.5,
                  arrow_length_ratio=0.12)
        ax.text(*(v * 1.05), f"axis {k + 1}", color=colors[k], fontsize=9)
    ax.scatter([0], [0], [0], color="black", s=40, label="centre of mass")

    ax.set_title(f"Principal axes of inertia (1fqy)\nRg = {mol.radius_of_gyration:.1f} Å")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_rmsf_profile(path: Path) -> None:
    """Per-residue (CA) RMSF across the NMR ensemble."""
    models = ms.read_pdb_models(DATA / "1aml.pdb")
    ca = [m.alpha_carbons() for m in models]
    rmsf = ms.ensemble.rmsf(ca)
    resids = ca[0].resids

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(resids, rmsf, color="#4363d8", linewidth=1.5)
    ax.fill_between(resids, rmsf, color="#4363d8", alpha=0.15)
    ax.set_title("Per-residue RMSF across 1aml NMR models")
    ax.set_xlabel("residue number")
    ax.set_ylabel("Cα RMSF (Å)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_principal_axes(OUT / "1fqy-principal-axes.png")
    render_rmsf_profile(OUT / "1aml-rmsf-profile.png")


if __name__ == "__main__":
    main()
