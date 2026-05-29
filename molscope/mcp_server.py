"""A Model Context Protocol (MCP) server that exposes MolScope to AI assistants.

This wraps MolScope's existing analysis features as MCP *tools* so an assistant
such as Claude Code or Claude Desktop can drive them in natural language: load a
structure (local file or RCSB id), compute descriptor tables, assign secondary
structure, build contact maps, find binding sites, summarise a molecular graph,
coarse-grain, and render PNG figures.

It adds no new science. Every tool is a thin, faithful adapter over the public
``molscope`` API documented in the user guide, returning JSON text (so results
are easy for a model to read) or a PNG image for the render tools.

Run it over stdio, which is how local MCP clients launch a server::

    molscope-mcp            # console script (needs the ``mcp`` extra)
    python -m molscope.mcp_server

Install the optional dependency with ``pip install "molscope[mcp]"``. Register it
with a client by pointing the client at the ``molscope-mcp`` command; for Claude
Code that is ``claude mcp add molscope -- molscope-mcp``.
"""

from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .molecule import Molecule

# Cap on how many per-item rows a tool will inline before truncating, so a large
# structure cannot flood the model's context with thousands of residues/pairs.
_MAX_ROWS = 2000


def _load(source: str) -> Molecule:
    """Resolve ``source`` to a :class:`~molscope.molecule.Molecule`.

    ``source`` is either a path to a local coordinate file (``.pdb``, ``.cif``,
    ``.xyz``, ``.sdf``, optionally gzipped) or a 4-character RCSB PDB id, which
    is fetched and cached.
    """
    from .io import fetch, read

    if os.path.exists(source):
        return read(source)
    token = source.strip()
    if len(token) == 4 and token.isalnum():
        return fetch(token)
    raise FileNotFoundError(
        f"{source!r} is neither an existing file nor a 4-character PDB id; "
        "pass a path like 'examples/data/1ubq.pdb' or an id like '1ubq'"
    )


def _three_state(code: str) -> str:
    return {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}.get(code, "C")


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays into plain JSON-serialisable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_server():  # noqa: C901 - a flat list of small tool adapters reads clearly
    """Construct and return the configured :class:`FastMCP` server.

    Imported lazily so ``import molscope.mcp_server`` works even when the ``mcp``
    extra is absent; only building/running the server needs it.
    """
    from mcp.server.fastmcp import FastMCP, Image

    server = FastMCP("molscope")

    @server.tool()
    def summarize_structure(source: str) -> str:
        """Load a structure and return a one-line summary.

        ``source`` is a local coordinate-file path or a 4-character PDB id.
        The summary reports atom count, formula, chains, and bounding-box size.
        """
        return _load(source).summary()

    @server.tool()
    def compute_descriptors(sources: list[str], preset: Optional[str] = None) -> str:
        """Compute MolScope's fixed-width structural descriptors for one or more structures.

        ``sources`` is a list of file paths and/or PDB ids. ``preset`` selects a
        descriptor preset (omit for the default set). Returns JSON with the
        ordered ``feature_names`` and one ``rows`` entry per source. This is the
        batch tool: pass several structures to get a comparable descriptor table.
        """
        # Compute per-structure (rather than via featurize_many) so a mix of file
        # paths and fetched PDB ids works through the same _load resolution.
        kwargs = {} if preset is None else {"preset": preset}
        names: Optional[list[str]] = None
        rows = []
        for src in sources:
            values = _load(src).descriptors(**kwargs)
            if names is None:
                names = list(values.keys())
            rows.append({"source": src, "values": [_jsonable(values[n]) for n in names]})
        return json.dumps(
            {"feature_names": names or [], "n_features": len(names or []), "rows": rows},
            indent=2,
        )

    @server.tool()
    def secondary_structure(source: str) -> str:
        """Assign protein secondary structure with MolScope's simplified DSSP.

        Returns JSON with per-residue codes (8-state DSSP letters) and a 3-state
        helix/strand/coil composition summary. Needs backbone N/CA/C/O atoms, so
        use a protein read from PDB/mmCIF.
        """
        ss = _load(source).secondary_structure()
        codes = ss.codes.tolist()
        resids = ss.resids.tolist()
        residues = [
            {"chain": c, "resid": int(r), "resname": rn, "code": code}
            for c, r, rn, code in zip(ss.chains, resids, ss.resnames, codes)
        ]
        total = len(codes) or 1
        helix = sum(1 for c in codes if _three_state(c) == "H")
        strand = sum(1 for c in codes if _three_state(c) == "E")
        coil = total - helix - strand
        truncated = len(residues) > _MAX_ROWS
        return json.dumps(
            {
                "n_residues": len(codes),
                "composition": {
                    "helix": helix,
                    "strand": strand,
                    "coil": coil,
                    "helix_fraction": helix / total,
                    "strand_fraction": strand / total,
                    "coil_fraction": coil / total,
                },
                "residues": residues[:_MAX_ROWS],
                "residues_truncated": truncated,
            },
            indent=2,
        )

    @server.tool()
    def contact_map(
        source: str,
        cutoff: float = 8.0,
        level: str = "residue",
        method: str = "ca",
        min_seq_sep: int = 0,
    ) -> str:
        """Build a contact map and return its summary plus contacting pairs.

        ``level`` is ``"residue"`` or ``"atom"``; for residue level ``method`` is
        ``"ca"``, ``"com"`` or ``"min"``. ``min_seq_sep`` drops same-chain
        contacts closer than that many sequence positions. Returns JSON with the
        contact count, contact order, and labelled contacting pairs (truncated if
        very large). The full dense matrix is intentionally not inlined.
        """
        cmap = _load(source).contact_map(
            cutoff=cutoff, level=level, method=method, min_seq_sep=min_seq_sep
        )
        labels = list(cmap.labels)
        pairs_idx = np.argwhere(np.triu(cmap.matrix, 1) > 0)
        pairs = [
            [labels[i] if i < len(labels) else int(i), labels[j] if j < len(labels) else int(j)]
            for i, j in pairs_idx.tolist()
        ]
        truncated = len(pairs) > _MAX_ROWS
        return json.dumps(
            {
                "level": cmap.level,
                "cutoff": cmap.cutoff,
                "method": method,
                "n_labels": len(labels),
                "n_contacts": cmap.n_contacts,
                "contact_order": cmap.contact_order(),
                "pairs": pairs[:_MAX_ROWS],
                "pairs_truncated": truncated,
            },
            indent=2,
        )

    @server.tool()
    def binding_site(source: str, ligand: Optional[str] = None, cutoff: float = 4.5) -> str:
        """Find protein residues around a bound ligand.

        ``ligand`` is a HETATM residue name (e.g. ``"BEN"``); omit it to use the
        single non-solvent ligand automatically. ``cutoff`` is the contact
        distance in angstrom. Returns JSON with the ligand and the binding-site
        residues ordered closest-first, each with its minimum distance.
        """
        from .contacts import binding_site as _binding_site

        site = _binding_site(_load(source), ligand=ligand, cutoff=cutoff)
        residues = [
            {
                "chain": res.chain,
                "resid": int(res.resid),
                "resname": res.resname,
                "min_distance": round(float(dist), 3),
            }
            for res, dist in zip(site.residues, site.min_distances)
        ]
        return json.dumps(
            {
                "ligand": str(site.ligand),
                "cutoff": site.cutoff,
                "n_residues": len(residues),
                "n_atom_contacts": site.n_atom_contacts,
                "residues": residues,
            },
            indent=2,
        )

    @server.tool()
    def molecular_graph(
        source: str, preset: str = "default", include_chemical_features: bool = False
    ) -> str:
        """Summarise the atom/bond molecular graph MolScope would export for ML.

        Returns JSON with node and edge counts, the node-feature matrix shape,
        and the ordered node/edge feature names for ``preset``. Set
        ``include_chemical_features=True`` to attach RDKit-backed aromatic flags
        (needs the ``chem`` extra). This describes the graph; use the Python API
        or CLI to export the actual PyG/DGL/NetworkX object.
        """
        from .graph import edge_feature_names, node_feature_names

        graph = _load(source).to_graph(include_chemical_features=include_chemical_features)
        node_matrix = graph.node_features(preset)
        return json.dumps(
            {
                "n_nodes": int(graph.n_atoms),
                "n_edges": int(graph.n_bonds),
                "preset": preset,
                "node_feature_matrix_shape": list(node_matrix.shape),
                "node_feature_names": list(node_feature_names(preset)),
                "edge_feature_names": list(edge_feature_names(preset)),
            },
            indent=2,
        )

    @server.tool()
    def coarse_grain(source: str, mapping: str = "residue_com") -> str:
        """Coarse-grain a structure to beads and report the assignment.

        ``mapping`` is ``"residue_com"``, ``"residue_centroid"`` or ``"martini"``.
        Returns JSON with the bead count and the number of atoms assigned versus
        dropped. This is a mapping for inspection, not a force-field model.
        """
        beads, report = _load(source).coarse_grain(mapping=mapping, return_report=True)
        return json.dumps(
            {
                "mapping": report.mapping,
                "n_beads": report.n_beads,
                "n_bonds": len(report.bonds),
                "n_dropped_atoms": len(report.dropped_atoms),
                "n_virtual_sites": len(report.virtual_sites),
                "summary": beads.summary(),
            },
            indent=2,
        )

    @server.tool()
    def render_structure(source: str, color_by: str = "element"):
        """Render the structure in 3D and return a PNG image.

        ``color_by`` is ``"element"``, ``"chain"``, ``"residue"`` or ``"ss"``
        (secondary structure). Returns a PNG of the 3D scatter view.
        """
        import matplotlib

        matplotlib.use("Agg")
        mol = _load(source)
        ax = mol.plot(color_by=color_by, show=False)
        return _png(ax.figure)

    @server.tool()
    def render_contact_map(
        source: str, cutoff: float = 8.0, level: str = "residue", method: str = "ca"
    ):
        """Render a contact map as a PNG heatmap.

        Same ``cutoff``/``level``/``method`` options as the ``contact_map`` tool.
        """
        import matplotlib

        matplotlib.use("Agg")
        cmap = _load(source).contact_map(cutoff=cutoff, level=level, method=method)
        ax = cmap.plot(show=False)
        return _png(ax.figure)

    def _png(figure) -> Image:
        import matplotlib.pyplot as plt

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        return Image(data=buffer.getvalue(), format="png")

    return server


def main() -> None:
    """Console-script entry point: build the server and serve over stdio."""
    try:
        server = build_server()
    except ImportError as exc:
        raise SystemExit(
            "The MolScope MCP server needs the 'mcp' package. "
            "Install it with: pip install 'molscope[mcp]'"
        ) from exc
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
