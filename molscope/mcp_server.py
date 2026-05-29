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
import math
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .molecule import Molecule

# Cap on how many per-item rows a tool will inline before truncating, so a large
# structure cannot flood the model's context with thousands of residues/pairs.
_MAX_ROWS = 2000


def _load(source: str, bond_perception: str = "geometric") -> Molecule:
    """Resolve ``source`` to a :class:`~molscope.molecule.Molecule`.

    ``source`` is either a path to a local coordinate file (``.pdb``, ``.cif``,
    ``.xyz``, ``.sdf``, optionally gzipped) or a 4-character RCSB PDB id, which
    is fetched and cached. ``bond_perception="template"`` attaches RDKit
    residue-template bonds (PDB only; see :func:`molscope.read_pdb`).
    """
    from .io import fetch, read

    if os.path.exists(source):
        return read(source, bond_perception=bond_perception)
    token = source.strip()
    if len(token) == 4 and token.isalnum():
        return fetch(token, bond_perception=bond_perception)
    raise FileNotFoundError(
        f"{source!r} is neither an existing file nor a 4-character PDB id; "
        "pass a path like 'examples/data/1ubq.pdb' or an id like '1ubq'"
    )


def _three_state(code: str) -> str:
    return {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}.get(code, "C")


def _num(value) -> Optional[float]:
    """A JSON-safe float: ``None`` for NaN/inf (invalid JSON), else ``float``."""
    f = float(value)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays into plain, JSON-safe Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float):
        return _num(value)
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
    def geometry(source: str) -> str:
        """Report whole-structure geometry: size, mass distribution, shape.

        Returns JSON with atom count, formula, chains, centre of mass, radius of
        gyration, bounding-box dimensions, and the principal moments of inertia.
        """
        mol = _load(source)
        return json.dumps(
            {
                "n_atoms": len(mol),
                "formula": mol.formula,
                "chains": sorted(set(mol.chain_ids())),
                "center_of_mass": [_num(v) for v in mol.center_of_mass.tolist()],
                "radius_of_gyration": _num(mol.radius_of_gyration),
                "dimensions": [_num(v) for v in np.asarray(mol.dimensions).tolist()],
                "principal_moments": [_num(v) for v in mol.principal_moments().tolist()],
            },
            indent=2,
        )

    @server.tool()
    def measure(source: str, atoms: list[int]) -> str:
        """Measure a geometric quantity between atoms by 0-based index.

        Pass 2 atom indices for a distance (angstrom), 3 for an angle (degrees),
        or 4 for a dihedral (degrees).
        """
        mol = _load(source)
        if len(atoms) == 2:
            return json.dumps({"kind": "distance", "atoms": atoms,
                               "value": _num(mol.distance(*atoms)), "unit": "angstrom"})
        if len(atoms) == 3:
            return json.dumps({"kind": "angle", "atoms": atoms,
                               "value": _num(mol.angle(*atoms)), "unit": "degrees"})
        if len(atoms) == 4:
            return json.dumps({"kind": "dihedral", "atoms": atoms,
                               "value": _num(mol.dihedral(*atoms)), "unit": "degrees"})
        raise ValueError("atoms must hold 2 (distance), 3 (angle), or 4 (dihedral) indices")

    @server.tool()
    def rmsd(source_a: str, source_b: str, align: bool = True) -> str:
        """Root-mean-square deviation between two structures with the same atom count.

        With ``align`` (default), the structures are Kabsch-superposed first so the
        result is the minimal RMSD; set it false for the RMSD as-is. Returns the
        value in angstrom.
        """
        a, b = _load(source_a), _load(source_b)
        return json.dumps({"rmsd": _num(a.rmsd(b, align=align)), "aligned": align,
                           "n_atoms": len(a), "unit": "angstrom"})

    @server.tool()
    def list_ligands(source: str, exclude_water: bool = True, exclude_ions: bool = True) -> str:
        """List the non-polymer (HETATM) groups in a structure.

        Useful before ``binding_site`` to see which ligand names are present.
        Returns JSON with each group's residue name, chain, residue id, and atom
        count. Waters and monatomic ions are excluded by default.
        """
        ligs = _load(source).ligands(exclude_water=exclude_water, exclude_ions=exclude_ions)
        return json.dumps(
            {
                "n_ligands": len(ligs),
                "ligands": [
                    {"resname": lig.resname, "chain": lig.chain, "resid": int(lig.resid),
                     "n_atoms": len(lig)}
                    for lig in ligs
                ],
            },
            indent=2,
        )

    @server.tool()
    def chain_interfaces(
        source: str, chain_a: Optional[str] = None, chain_b: Optional[str] = None,
        cutoff: float = 5.0,
    ) -> str:
        """Analyse inter-chain contacts.

        With both ``chain_a`` and ``chain_b``, return the residues on each side of
        that interface (within ``cutoff`` angstrom) and the atom-contact count.
        With neither, return the all-pairs chain contact matrix instead.
        """
        mol = _load(source)
        if chain_a and chain_b:
            iface = mol.interface(chain_a, chain_b, cutoff=cutoff)

            def fmt(residues):
                return [{"chain": r.chain, "resid": int(r.resid), "resname": r.resname}
                        for r in residues]

            return json.dumps(
                {"chain_a": iface.chain_a, "chain_b": iface.chain_b, "cutoff": cutoff,
                 "n_atom_contacts": iface.n_atom_contacts,
                 "residues_a": fmt(iface.residues_a), "residues_b": fmt(iface.residues_b)},
                indent=2,
            )
        ccm = mol.chain_contacts(cutoff=cutoff)
        return json.dumps(
            {"cutoff": cutoff, "chains": list(ccm.chains),
             "contact_matrix": [[int(v) for v in row] for row in ccm.matrix.tolist()]},
            indent=2,
        )

    @server.tool()
    def backbone_torsions(source: str) -> str:
        """Per-residue backbone dihedral angles (Ramachandran phi/psi/omega).

        Returns JSON with one entry per residue in chain/residue order. Angles are
        ``null`` where undefined (phi at a chain start, psi/omega at a chain end).
        """
        bt = _load(source).backbone_torsions()
        residues = [
            {"chain": c, "resid": int(r),
             "phi": _num(phi), "psi": _num(psi), "omega": _num(omega)}
            for c, r, phi, psi, omega in zip(
                bt.chains, bt.resids.tolist(), bt.phi.tolist(), bt.psi.tolist(), bt.omega.tolist()
            )
        ]
        return json.dumps(
            {"n_residues": len(residues), "residues": residues[:_MAX_ROWS],
             "residues_truncated": len(residues) > _MAX_ROWS},
            indent=2,
        )

    @server.tool()
    def ensemble_summary(source: str) -> str:
        """Summarise a multi-model (e.g. NMR) ensemble.

        Reads every model from a multi-model PDB and returns JSON with the model
        count, mean/max pairwise RMSD across models, a per-atom RMSF summary, and
        the number of conformational clusters. Errors on single-model inputs.
        """
        from .ensemble import cluster, rmsd_matrix, rmsf
        from .io import read_pdb_models

        models = read_pdb_models(source)
        if len(models) < 2:
            raise ValueError("ensemble_summary needs a multi-model file (e.g. an NMR PDB)")
        mat = rmsd_matrix(models)
        upper = mat[np.triu_indices_from(mat, k=1)]
        fluct = rmsf(models)
        return json.dumps(
            {
                "n_models": len(models),
                "mean_pairwise_rmsd": _num(upper.mean()),
                "max_pairwise_rmsd": _num(upper.max()),
                "rmsf_mean": _num(fluct.mean()),
                "rmsf_max": _num(fluct.max()),
                "n_clusters": cluster(models).n_clusters,
                "unit": "angstrom",
            },
            indent=2,
        )

    @server.tool()
    def chemical_features(source: str, bond_perception: str = "template") -> str:
        """RDKit-perceived per-atom chemistry (needs the ``chem`` extra).

        Returns JSON with the formal-charge sum, the number of aromatic atoms and
        bonds, and the atom/bond counts RDKit assigned after sanitisation.

        ``bond_perception`` defaults to ``"template"``, which uses RDKit's
        residue-aware PDB reader so standard-residue proteins get correct bond
        orders and aromatic rings. (Plain distance-based ``"geometric"`` bonds
        miss all of that on bare PDBs.) Template perception applies to PDB inputs
        only; other formats fall back to their explicit/geometric bonds.
        """
        bp = bond_perception
        if bp == "template" and os.path.exists(source) and not source.lower().endswith(
            (".pdb", ".pdb.gz", ".ent")
        ):
            bp = "geometric"  # templates apply to PDB only; SDF/MOL carry real bonds
        feats = _load(source, bond_perception=bp).chemical_features()
        return json.dumps(
            {
                "n_atoms": int(len(feats.formal_charges)),
                "total_formal_charge": int(sum(int(c) for c in feats.formal_charges)),
                "n_aromatic_atoms": int(sum(bool(a) for a in feats.aromatic_atoms)),
                "n_bonds": int(len(feats.bond_orders)),
                "n_aromatic_bonds": int(sum(bool(a) for a in feats.aromatic_bonds)),
            },
            indent=2,
        )

    @server.tool()
    def validate_cif(source: str) -> str:
        """Validate an mmCIF/CIF file (needs the ``cif`` extra / gemmi).

        Returns JSON with whether the file is valid, syntax/atom-site status, block
        and atom-row counts, and any errors or warnings.
        """
        from .cif import validate_cif as _validate

        report = _validate(source)
        return json.dumps(
            {
                "path": report.path, "valid": report.valid,
                "syntax_ok": report.syntax_ok, "atom_site_ok": report.atom_site_ok,
                "n_blocks": report.n_blocks, "n_atom_site_rows": report.n_atom_site_rows,
                "dictionary_checked": report.dictionary_checked,
                "errors": list(report.errors), "warnings": list(report.warnings),
            },
            indent=2,
        )

    @server.tool()
    def select_diverse(
        table: str, n: int, descriptor_cols: Optional[list[str]] = None,
        smiles_col: Optional[str] = None, compute_descriptors: bool = False,
    ) -> str:
        """Pick a diverse subset of molecules from a CSV/XLSX table.

        ``table`` is a path to a ``.csv`` or ``.xlsx`` file of molecules. Select on
        existing numeric columns via ``descriptor_cols`` (e.g. ``["MW", "ALogP"]``),
        or set ``compute_descriptors`` with ``smiles_col`` to compute RDKit
        descriptors (``MolLogP`` is the ALogP equivalent) and select on those.
        Returns the chosen rows by MaxMin (farthest-first) diversity selection.
        """
        from .library import read_table, smiles_descriptors
        from .library import select_diverse as _pick

        tab = read_table(table)
        if compute_descriptors:
            if not smiles_col:
                raise ValueError("compute_descriptors needs smiles_col")
            matrix, names = smiles_descriptors(tab.column(smiles_col))
            tab = tab.with_columns(names, matrix)
        elif descriptor_cols:
            names, matrix = list(descriptor_cols), tab.numeric_matrix(descriptor_cols)
        else:
            raise ValueError("provide descriptor_cols, or compute_descriptors with smiles_col")
        chosen = tab.select_rows(_pick(matrix, n))
        return json.dumps(
            {"selected": len(chosen), "of": len(tab), "descriptors": names,
             "rows": [dict(r) for r in chosen.rows]},
            indent=2, default=_jsonable,
        )

    @server.tool()
    def render_structure(source: str, color_by: str = "element", save_path: Optional[str] = None):
        """Render the structure in 3D.

        ``color_by`` is ``"element"``, ``"chain"``, ``"residue"`` or ``"ss"``
        (secondary structure). Pass ``save_path`` (e.g. ``"~/Desktop/view.png"``)
        to write the figure to a file and return its path; omit it to return the
        image inline. The format follows the ``save_path`` extension
        (``.png``/``.pdf``/``.svg``), defaulting to PNG.
        """
        import matplotlib

        matplotlib.use("Agg")
        mol = _load(source)
        ax = mol.plot(color_by=color_by, show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool()
    def render_contact_map(
        source: str, cutoff: float = 8.0, level: str = "residue", method: str = "ca",
        save_path: Optional[str] = None,
    ):
        """Render a contact map as a heatmap.

        Same ``cutoff``/``level``/``method`` options as the ``contact_map`` tool.
        Pass ``save_path`` to write the figure to a file and return its path;
        omit it to return the image inline.
        """
        import matplotlib

        matplotlib.use("Agg")
        cmap = _load(source).contact_map(cutoff=cutoff, level=level, method=method)
        ax = cmap.plot(show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool()
    def render_distance_matrix(source: str, save_path: Optional[str] = None):
        """Render the dense pairwise atom-distance matrix as a heatmap.

        Pass ``save_path`` to write the figure to a file and return its path;
        omit it to return the image inline.
        """
        import matplotlib

        matplotlib.use("Agg")
        ax = _load(source).plot_distance_matrix(show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool()
    def render_rmsd_heatmap(source: str, save_path: Optional[str] = None):
        """Render a multi-model ensemble's pairwise-RMSD matrix as a heatmap.

        ``source`` must be a multi-model (e.g. NMR) PDB. Pass ``save_path`` to
        write the figure to a file and return its path; omit it for inline.
        """
        import matplotlib

        matplotlib.use("Agg")
        from .ensemble import rmsd_matrix
        from .io import read_pdb_models
        from .plotting import plot_rmsd_heatmap

        models = read_pdb_models(source)
        if len(models) < 2:
            raise ValueError("render_rmsd_heatmap needs a multi-model file (e.g. an NMR PDB)")
        ax = plot_rmsd_heatmap(rmsd_matrix(models), show=False)
        return _figure_result(ax.figure, save_path)

    def _figure_result(figure, save_path: Optional[str]):
        """Save the figure to ``save_path`` (returning the path) or return it inline.

        When ``save_path`` is given the figure is written to disk and the absolute
        path is returned as text, so the user gets a real file to open or share.
        The image format follows the path extension (png/pdf/svg/jpg/tiff),
        defaulting to PNG. Otherwise the PNG is returned inline.
        """
        import matplotlib.pyplot as plt

        if save_path:
            path = os.path.abspath(os.path.expanduser(save_path))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            fmt = ext if ext in {"png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff"} else "png"
            figure.savefig(path, format=fmt, dpi=150, bbox_inches="tight")
            plt.close(figure)
            return f"Saved figure to {path}"

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
