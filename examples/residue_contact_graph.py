"""Build and draw a residue contact graph.

Run it from the repository root:

    uv run python examples/residue_contact_graph.py

The script writes ``docs/assets/graphs/1fqy-residue-contact-graph.png`` and
prints the graph counts plus a sample node/edge. It uses NetworkX only for the
layout/drawing step; the ``ResidueContactGraph`` itself is dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "graphs" / "1fqy-residue-contact-graph.png"

RESIDUE_GROUP_COLORS = {
    "hydrophobic": "#2F6F73",
    "polar": "#7A9E3A",
    "positive": "#C75C5C",
    "negative": "#5D6FB8",
    "special": "#8C6BB1",
}


def _require_networkx():
    try:
        import matplotlib
        import networkx as nx

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "Install NetworkX for this drawing example:\n"
            "  pip install 'molscope[graph]'"
        ) from exc
    return nx, plt


def _project_to_plane(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vh[:2].T
    span = np.ptp(xy, axis=0)
    span[span == 0.0] = 1.0
    return xy / span


def _residue_group(resname: str) -> str:
    resname = resname.upper()
    if resname in {"ASP", "GLU"}:
        return "negative"
    if resname in {"ARG", "HIS", "LYS"}:
        return "positive"
    if resname in {"ASN", "GLN", "SER", "THR", "TYR"}:
        return "polar"
    if resname in {"GLY", "PRO", "CYS"}:
        return "special"
    return "hydrophobic"


def main(output: Path = DEFAULT_OUTPUT):
    nx, plt = _require_networkx()

    mol = ms.read(DATA / "1fqy.pdb")
    graph = mol.to_residue_contact_graph(cutoff=8.0, method="ca", min_seq_sep=4)
    G = graph.to_networkx()

    xy = _project_to_plane(graph.coords)
    pos = {i: xy[i] for i in range(graph.n_residues)}
    colors = [
        RESIDUE_GROUP_COLORS[_residue_group(G.nodes[i]["resname"])]
        for i in range(graph.n_residues)
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#59616F",
        alpha=0.18,
        width=0.7,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=34,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_title("1FQY residue contact graph: CA contacts within 8 A")
    ax.set_axis_off()
    fig.tight_layout(pad=0.4)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {output}")
    print(f"Residues: {graph.n_residues}")
    print(f"Contacts: {graph.n_contacts}")
    print(f"Node 0: {G.nodes[0]}")
    first_edge = next(iter(G.edges))
    print(f"First edge {first_edge}: {G.edges[first_edge]}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    main(path)
