"""Generate notebooks/protein_analysis_from_scratch.ipynb.

The notebook is a teaching-oriented companion to examples/protein_analysis.py.
Keeping it generated avoids fragile hand edits to notebook JSON.
"""

from __future__ import annotations

import json
from pathlib import Path


def _lines(src):
    text = "\n".join(src)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(*src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(src),
    }


cells = [
    md(
        "# Protein Analysis From Scratch",
        "",
        "This notebook treats protein structures as structured coordinate data.",
        "It uses the three bundled PDB files:",
        "",
        "- `1fqy.pdb`: aquaporin-1, for backbone atoms, alpha carbons, contact",
        "  maps and helix-rich secondary structure.",
        "- `1aml.pdb`: a 20-model NMR ensemble, for contact-frequency analysis.",
        "- `3ptb.pdb`: trypsin with benzamidine, waters and calcium, for ligand",
        "  and binding-site analysis.",
        "",
        "By the end you should be able to find residues, chains, ligands,",
        "waters, alpha carbons, contact maps, binding-site residues and a",
        "simplified secondary-structure assignment.",
    ),
    md(
        "## Outline",
        "",
        "1. Load PDB structures and inspect metadata.",
        "2. Select backbone atoms and alpha carbons.",
        "3. Build residue contact maps.",
        "4. Compare contact frequencies across an NMR ensemble.",
        "5. Detect ligands, waters and binding-site residues.",
        "6. Read MolScope's simplified DSSP-style secondary structure.",
    ),
    code(
        "from pathlib import Path",
        "",
        "import matplotlib.pyplot as plt",
        "import molscope as ms",
        "",
        "# Works from the repo root or from notebooks/.",
        "DATA = Path('examples/data')",
        "if not DATA.exists():",
        "    DATA = Path('..') / 'examples' / 'data'",
        "",
        "structures = {",
        "    '1fqy': ms.read(DATA / '1fqy.pdb'),",
        "    '1aml_first_model': ms.read(DATA / '1aml.pdb'),",
        "    '3ptb': ms.read(DATA / '3ptb.pdb'),",
        "}",
        "",
        "for name, mol in structures.items():",
        "    residues = sum(1 for _ in mol.residue_groups())",
        "    print(",
        "        f'{name:16} {len(mol):4d} atoms | '",
        "        f'{residues:3d} residues | chains {mol.chain_ids()}'",
        "    )",
    ),
    md(
        "## 1. Proteins as structured coordinate data",
        "",
        "A PDB-backed `Molecule` is more than an `(N, 3)` coordinate array.",
        "MolScope also preserves atom names, residue names, residue ids, chain",
        "ids and ATOM/HETATM records. Those fields are what make protein-specific",
        "selection possible.",
    ),
    code(
        "for name, mol in structures.items():",
        "    water_atoms = len(mol.select(resname='HOH')) if 'HOH' in set(mol.resnames) else 0",
        "    ligands = mol.ligands()",
        "    print(f'\\n{name}')",
        "    print(",
        "        '  first atom:',",
        "        mol.atom_names[0], mol.resnames[0], mol.resids[0], mol.chains[0],",
        "    )",
        "    print('  protein atoms:', len(mol.protein()))",
        "    print('  hetero atoms :', len(mol.hetero_atoms()))",
        "    print('  water atoms  :', water_atoms)",
        "    print('  ligands      :', ligands)",
    ),
    md(
        "## 2. Backbone atoms and alpha carbons",
        "",
        "For proteins, the backbone atom names are usually `N`, `CA`, `C` and",
        "`O`. Alpha carbons (`CA`) are one common residue-level proxy for",
        "distances, RMSD and contact maps.",
    ),
    code(
        "aqp = structures['1fqy']",
        "",
        "backbone = aqp.backbone()",
        "ca = aqp.alpha_carbons()",
        "",
        "print(backbone.summary())",
        "print(ca.summary())",
        "print('first five CA residue labels:')",
        "for atom_name, resname, resid, chain in zip(",
        "    ca.atom_names[:5], ca.resnames[:5], ca.resids[:5], ca.chains[:5]",
        "):",
        "    print(f'  {chain}:{resname}{resid} atom={atom_name}')",
    ),
    md(
        "## 3. Residue contact maps",
        "",
        "A residue contact map answers: which residue pairs are close in 3D?",
        "Here we use CA-CA contacts within 8 A and drop sequence-local contacts",
        "with `min_seq_sep=4` so the map emphasizes tertiary contacts.",
    ),
    code(
        "cmap = aqp.contact_map(cutoff=8.0, level='residue', method='ca', min_seq_sep=4)",
        "print('matrix shape:', cmap.matrix.shape)",
        "print('non-local contacts:', cmap.n_contacts)",
        "print('relative contact order:', round(cmap.contact_order(), 3))",
        "",
        "ax = cmap.plot(show=False)",
        "ax.set_title('1FQY residue contact map')",
        "plt.show()",
    ),
    md(
        "## 4. NMR ensemble contact frequency",
        "",
        "`1aml.pdb` contains 20 NMR models. A contact-frequency map reports how",
        "often each residue pair is in contact across those models: `1.0` means",
        "always in contact, intermediate values mark conformational variability.",
    ),
    code(
        "models = ms.read_pdb_models(DATA / '1aml.pdb')",
        "freq = ms.ensemble_contact_frequency(models, cutoff=8.0)",
        "radii = [model.radius_of_gyration for model in models]",
        "",
        "print('models:', len(models))",
        "print('frequency matrix:', freq.matrix.shape)",
        "print('pairs observed at least once:', freq.n_contacts)",
        "print(f'radius of gyration range: {min(radii):.2f}-{max(radii):.2f} A')",
        "",
        "ax = freq.plot(show=False)",
        "ax.set_title('1AML contact frequency across NMR models')",
        "plt.show()",
    ),
    md(
        "## 5. Ligands, waters and binding-site residues",
        "",
        "`3ptb.pdb` includes trypsin, benzamidine (`BEN`), waters and a calcium",
        "ion. `ligands()` filters out waters and common ions by default;",
        "`binding_site()` then reports protein residues around the selected ligand.",
    ),
    code(
        "trypsin = structures['3ptb']",
        "waters = trypsin.select(resname='HOH')",
        "site = trypsin.binding_site(cutoff=4.5)",
        "",
        "print('hetero atoms:', len(trypsin.hetero_atoms()))",
        "print('water atoms:', len(waters))",
        "print('ligands:', trypsin.ligands())",
        "print(site)",
        "print('closest residues:')",
        "for residue, distance in zip(site.residues[:8], site.min_distances[:8]):",
        "    print(f'  {residue!s:<10} {distance:.2f} A')",
    ),
    md(
        "## 6. Secondary structure basics",
        "",
        "MolScope includes a simplified, dependency-free DSSP-style assignment.",
        "It uses backbone N/CA/C/O atoms and hydrogen-bond patterns to assign",
        "8-state codes, then can reduce them to helix/strand/coil.",
        "",
        "**Important:** this is educational/prototyping DSSP-style analysis, not",
        "a bit-identical replacement for canonical `mkdssp`. Use reference DSSP",
        "when production-grade secondary-structure labels are required.",
    ),
    code(
        "for name in ['1fqy', '3ptb']:",
        "    ss = structures[name].secondary_structure()",
        "    summary = ss.summary()",
        "    print(f'\\n{name}')",
        "    print('  residues assigned:', len(ss))",
        "    print('  summary:', summary)",
        "    print('  first simplified codes:', ss.simplified()[:60])",
        "    print('  first elements:', ss.segments()[:4])",
    ),
    md(
        "## Exercise: change the contact definition",
        "",
        "Try changing the residue contact method from CA distance to closest-atom",
        "distance. Before running the answer cell, predict whether the contact",
        "count should increase or decrease.",
    ),
    code(
        "ca_contacts = aqp.contact_map(cutoff=8.0, level='residue', method='ca', min_seq_sep=4)",
        "min_contacts = aqp.contact_map(cutoff=8.0, level='residue', method='min', min_seq_sep=4)",
        "",
        "print('CA contacts :', ca_contacts.n_contacts)",
        "print('min contacts:', min_contacts.n_contacts)",
        "print('difference  :', min_contacts.n_contacts - ca_contacts.n_contacts)",
    ),
    md(
        "Closest-atom contacts usually increase the count because two side chains",
        "can touch even when their alpha carbons are farther apart.",
        "",
        "Common pitfall: secondary structure and residue contact maps require",
        "PDB/mmCIF-style atom and residue metadata. Bare XYZ files carry",
        "coordinates and elements only, so they cannot answer residue-level",
        "protein questions without extra annotation.",
        "",
        "Script version: `examples/protein_analysis.py`.",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "protein_analysis_from_scratch.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {out} ({len(cells)} cells)")
