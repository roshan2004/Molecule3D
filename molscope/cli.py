"""Command-line entry point for MolScope.

Supports viewing single structures, batch analysis to CSV, and batch graph export.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from .io import fetch, read

_SELECTION_KEYS = {
    "element", "chain", "resname", "atom_name", "resid", "icode", "residue_id", "hetero",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="molscope",
        description="Lightweight molecular structure analysis and ML tools.",
    )
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # -- VIEW subcommand ---------------------------------------------------
    view_parser = subparsers.add_parser(
        "view", help="visualise a single structure (default)"
    )
    src = view_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="path to a structure file")
    src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")

    view_parser.add_argument(
        "--select", metavar="SPEC", action="append",
        help=(
            "atom selection; repeat or combine with 'and', e.g. "
            "'chain=A and atom_name=CA'"
        ),
    )
    view_parser.add_argument(
        "--color-by", choices=["element", "chain", "residue"], default="element",
    )
    view_parser.add_argument("--center", action="store_true", help="center at origin")
    view_parser.add_argument(
        "--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
        help="shift atoms by this vector",
    )
    view_parser.add_argument(
        "--rotate", nargs=2, metavar=("AXIS", "DEG"),
        help="rotate about AXIS (x/y/z) by DEG degrees",
    )
    bonds = view_parser.add_mutually_exclusive_group()
    bonds.add_argument("--bonds", dest="bonds", action="store_true", help="force bonds")
    bonds.add_argument("--no-bonds", dest="bonds", action="store_false", help="hide bonds")
    view_parser.set_defaults(bonds=None)

    view_parser.add_argument("--save", metavar="PATH", help="save figure to file")
    view_parser.add_argument("--gif", metavar="PATH", help="save spinning GIF")


    # -- ANALYZE subcommand ------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze", help="batch compute molecular descriptors"
    )
    analyze_parser.add_argument("files", nargs="+", help="files or glob patterns")
    analyze_parser.add_argument("--out", "-o", required=True, help="output CSV file")
    analyze_parser.add_argument(
        "--preset", choices=["native-basic", "native-3d", "rdkit-basic"],
        default="native-basic", help="descriptor preset"
    )
    analyze_parser.add_argument("--jobs", "-j", type=int, default=1, help="parallel jobs")

    # -- BINDING-SITE subcommand ------------------------------------------
    binding_parser = subparsers.add_parser(
        "binding-site", help="write ligand binding-site residue contacts to CSV"
    )
    src = binding_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="path to a protein-ligand structure file")
    src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")
    binding_parser.add_argument("--out", "-o", required=True, help="output residue CSV file")
    binding_parser.add_argument(
        "--cutoff", type=float, default=4.5,
        help="protein-ligand atom contact cutoff in angstrom"
    )
    binding_parser.add_argument(
        "--ligand",
        help="ligand residue name, or chain:resid[:icode] for a specific HETATM group",
    )
    binding_parser.add_argument(
        "--descriptors-out",
        help="optional one-row CSV of pocket-basic descriptors",
    )

    # -- EXPORT subcommand -------------------------------------------------
    export_parser = subparsers.add_parser(
        "export", help="batch export molecular graphs for ML"
    )
    export_parser.add_argument("files", nargs="+", help="files or glob patterns")
    export_parser.add_argument(
        "--to", choices=["pyg", "dgl", "nx"], required=True, help="target format"
    )
    export_parser.add_argument("--self-loops", action="store_true", help="add (i, i) edges")
    export_parser.add_argument("--global-node", action="store_true", help="add virtual master node")
    export_parser.add_argument(
        "--pe", choices=["laplacian", "random_walk"], help="add positional encodings"
    )
    export_parser.add_argument("--pe-k", type=int, default=8, help="PE dimension")
    export_parser.add_argument("--out-dir", "-o", required=True, help="output directory")

    export_parser.add_argument("--jobs", "-j", type=int, default=1, help="parallel jobs")

    # -- SELECT subcommand -------------------------------------------------
    select_parser = subparsers.add_parser(
        "select", help="pick a diverse subset from a molecule table (CSV/XLSX)"
    )
    select_parser.add_argument("file", help="input table (.csv, .tsv or .xlsx)")
    select_parser.add_argument(
        "--num", "-n", type=int, required=True, help="number of molecules to select"
    )
    select_parser.add_argument(
        "--out", "-o", help="write the selection to this .csv/.xlsx (default: print a summary)"
    )
    select_parser.add_argument(
        "--descriptor-cols", nargs="+", metavar="COL",
        help="existing numeric columns to select on, e.g. --descriptor-cols MW ALogP",
    )
    select_parser.add_argument(
        "--smiles-col", metavar="COL",
        help="column holding SMILES (use with --compute-descriptors)",
    )
    select_parser.add_argument(
        "--compute-descriptors", action="store_true",
        help="compute RDKit descriptors from --smiles-col and select on them",
    )
    select_parser.add_argument(
        "--rdkit-descriptors", nargs="+", metavar="NAME",
        help="which RDKit descriptors to compute (default: MolWt MolLogP TPSA ...)",
    )
    select_parser.add_argument(
        "--no-standardize", dest="standardize", action="store_false",
        help="select on raw descriptors instead of z-scored ones",
    )
    select_parser.set_defaults(standardize=True)

    # -- PREPARE subcommand ------------------------------------------------
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="build ML-ready train/validation/test splits from a table or SDF",
    )
    prepare_parser.add_argument("file", help="input dataset (.csv, .tsv, .xlsx or .sdf)")
    prepare_parser.add_argument(
        "--out-dir", "-o", default="prepared",
        help="output directory for splits, descriptors and report (default: prepared/)",
    )
    prepare_parser.add_argument(
        "--split", choices=["random", "diversity", "scaffold"], default="random",
        help="split strategy (scaffold needs RDKit)",
    )
    prepare_parser.add_argument(
        "--test", type=float, default=0.1, help="test fraction (default: 0.1)"
    )
    prepare_parser.add_argument(
        "--val", type=float, default=0.1, help="validation fraction (default: 0.1)"
    )
    prepare_parser.add_argument("--seed", type=int, default=0, help="random seed")
    prepare_parser.add_argument(
        "--smiles-col", metavar="COL",
        help="column holding SMILES (required for scaffold/dedup/descriptors on a table)",
    )
    prepare_parser.add_argument(
        "--descriptor-cols", nargs="+", metavar="COL",
        help="existing numeric columns to use for a diversity split",
    )
    prepare_parser.add_argument(
        "--compute-descriptors", action="store_true",
        help="compute RDKit descriptors from --smiles-col (needs RDKit)",
    )
    prepare_parser.add_argument(
        "--rdkit-descriptors", nargs="+", metavar="NAME",
        help="which RDKit descriptors to compute (default: MolWt MolLogP TPSA ...)",
    )
    prepare_parser.add_argument(
        "--dedup", choices=["none", "exact", "canonical"], default="none",
        help="drop duplicate molecules (canonical needs RDKit)",
    )
    prepare_parser.add_argument(
        "--fingerprints", action="store_true",
        help="add a Morgan fingerprint column (needs RDKit)",
    )
    prepare_parser.add_argument(
        "--no-standardize", dest="standardize", action="store_false",
        help="diversity split on raw descriptors instead of z-scored ones",
    )
    prepare_parser.add_argument(
        "--no-figure", dest="figure", action="store_false",
        help="skip writing report.png",
    )
    prepare_parser.set_defaults(standardize=True, figure=True)

    # Default to 'view' if no subcommand provided
    if argv is None:
        argv = sys.argv[1:]
    argv = _default_to_view(argv, subparsers.choices)

    args = parser.parse_args(argv)

    if args.command == "view":
        return _run_view(args)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "binding-site":
        return _run_binding_site(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "select":
        return _run_select(args)
    if args.command == "prepare":
        return _run_prepare(args)

    return 0


def _default_to_view(argv, subcommands) -> list[str]:
    if not argv:
        return ["view"]
    if argv[0] in subcommands or argv[0] in {"-h", "--help"}:
        return list(argv)
    return ["view"] + list(argv)


def _run_view(args: argparse.Namespace) -> int:
    mol = fetch(args.fetch) if args.fetch else read(args.file)
    if args.select:
        try:
            selection = _parse_selection(args.select)
        except ValueError as e:
            print(f"Invalid --select: {e}", file=sys.stderr)
            return 2
        try:
            mol = mol.select(**selection)
        except ValueError as e:
            print(f"Selection failed: {e}", file=sys.stderr)
            return 2
    if args.center:
        mol = mol.centered()
    if args.translate:
        mol = mol.translate(args.translate)
    if args.rotate:
        mol = mol.rotate(args.rotate[0], float(args.rotate[1]))

    print(mol.summary())

    if args.gif:
        from .plotting import spin_gif
        spin_gif(mol, args.gif, color_by=args.color_by, show_bonds=args.bonds)
        print(f"saved {args.gif}")
        return 0

    show = args.save is None
    ax = mol.plot(color_by=args.color_by, show_bonds=args.bonds, show=show)
    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    return 0


def _run_select(args: argparse.Namespace) -> int:
    from .library import read_table, select_diverse, smiles_descriptors

    if args.num <= 0:
        print("--num must be a positive integer", file=sys.stderr)
        return 2
    if args.compute_descriptors and not args.smiles_col:
        print("--compute-descriptors requires --smiles-col", file=sys.stderr)
        return 2
    if not args.compute_descriptors and not args.descriptor_cols:
        print(
            "provide --descriptor-cols COL [COL ...], or "
            "--compute-descriptors --smiles-col COL",
            file=sys.stderr,
        )
        return 2

    try:
        table = read_table(args.file)
    except (OSError, ValueError, ImportError) as exc:
        print(f"could not read {args.file}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.compute_descriptors:
            smiles = table.column(args.smiles_col)
            matrix, names = smiles_descriptors(smiles, names=args.rdkit_descriptors)
            table = table.with_columns(names, matrix)
        else:
            names = list(args.descriptor_cols)
            matrix = table.numeric_matrix(names)
    except (KeyError, ValueError, ImportError) as exc:
        print(f"descriptor error: {exc}", file=sys.stderr)
        return 2

    try:
        indices = select_diverse(matrix, args.num, standardize=args.standardize)
    except ValueError as exc:
        print(f"selection error: {exc}", file=sys.stderr)
        return 2

    selection = table.select_rows(indices)
    print(
        f"selected {len(selection)} of {len(table)} molecules "
        f"on {', '.join(names)} (diverse MaxMin)"
    )
    if len(selection) < args.num:
        print(
            f"note: only {len(selection)} rows had complete descriptors, "
            f"fewer than the requested {args.num}",
            file=sys.stderr,
        )

    if args.out:
        try:
            selection.write(args.out)
        except (OSError, ValueError, ImportError) as exc:
            print(f"could not write {args.out}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.out}")
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    from .prepare import prepare_dataset

    if not 0 <= args.test < 1 or not 0 <= args.val < 1 or args.test + args.val >= 1:
        print("--test and --val must be in [0, 1) and sum to < 1", file=sys.stderr)
        return 2

    try:
        dataset = prepare_dataset(
            args.file,
            smiles_col=args.smiles_col,
            descriptor_cols=args.descriptor_cols,
            compute_descriptors=args.compute_descriptors,
            rdkit_descriptors=args.rdkit_descriptors,
            split=args.split,
            test=args.test,
            val=args.val,
            seed=args.seed,
            standardize=args.standardize,
            dedup=args.dedup,
            fingerprints=args.fingerprints,
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        return 2

    try:
        written = dataset.write(args.out_dir, make_figure=args.figure)
    except (OSError, ValueError, ImportError) as exc:
        print(f"could not write to {args.out_dir}: {exc}", file=sys.stderr)
        return 2

    sizes = dataset.split.sizes
    print(
        f"prepared {dataset.n_prepared} of {dataset.n_input} molecules "
        f"({args.split} split): "
        f"train={sizes['train']} validation={sizes['validation']} test={sizes['test']}"
    )
    if dataset.n_duplicates:
        print(f"removed {dataset.n_duplicates} duplicate(s) ({args.dedup})")
    print(f"wrote {len(written)} files to {args.out_dir}/")
    return 0


def _parse_selection(specs) -> dict:
    """Parse CLI selection specs into ``Molecule.select`` keyword arguments."""
    if isinstance(specs, str):
        specs = [specs]

    selection = {}
    for spec in specs:
        parts = [
            part.strip()
            for part in re.split(r"\s+and\s+", spec.strip(), flags=re.IGNORECASE)
        ]
        for part in parts:
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"{part!r} is not key=value")
            key, value = [piece.strip() for piece in part.split("=", 1)]
            if not key or not value:
                raise ValueError(f"{part!r} is not key=value")
            if key not in _SELECTION_KEYS:
                supported = ", ".join(sorted(_SELECTION_KEYS))
                raise ValueError(f"unsupported field {key!r}; use one of: {supported}")
            if key in selection:
                raise ValueError(f"field {key!r} was specified more than once")
            selection[key] = _parse_selection_value(key, value)

    if not selection:
        raise ValueError("selection is empty")
    return selection


def _parse_selection_value(key: str, value: str):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    if key == "resid":
        try:
            if ":" in value:
                low, high = value.split(":", 1)
                return (int(low), int(high))
            if "-" in value and not value.startswith("-"):
                low, high = value.split("-", 1)
                return (int(low), int(high))
            return int(value)
        except ValueError as exc:
            raise ValueError(
                "resid expects an integer or inclusive range like 10-20"
            ) from exc

    if key == "residue_id":
        parts = value.split(":")
        if len(parts) not in (2, 3, 4):
            raise ValueError("residue_id expects chain:resid[:icode[:resname]]")
        try:
            resid = int(parts[1])
        except ValueError as exc:
            raise ValueError("residue_id selectors need an integer resid") from exc
        return tuple([parts[0], resid, *parts[2:]])

    if key == "hetero":
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "hetatm", "hetero"}:
            return True
        if lowered in {"0", "false", "no", "atom", "protein"}:
            return False
        raise ValueError("hetero expects true/false")

    return value



def _parse_ligand(value: str | None):
    if value is None:
        return None
    if ":" in value:
        parts = value.split(":")
        if len(parts) not in (2, 3, 4):
            raise ValueError("ligand selectors expect chain:resid[:icode[:resname]]")
        try:
            return tuple([parts[0], int(parts[1]), *parts[2:]])
        except ValueError as exc:
            raise ValueError("chain:resid ligand selectors need an integer resid") from exc
    return value


def _analyze_one(path: str, preset: str):
    """Compute flattened descriptors for one structure (worker; must be top-level
    so it is picklable under the ``spawn`` start method on macOS/Windows)."""
    from .descriptors import descriptors, flatten_descriptors

    try:
        mol = read(path)
        desc = descriptors(mol, preset=preset)
        return {"file": path, **flatten_descriptors(desc)}
    except Exception as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
        return None


def _run_analyze(args: argparse.Namespace) -> int:
    import csv
    from functools import partial
    from multiprocessing import Pool

    paths = _expand_globs(args.files)
    print(f"Analyzing {len(paths)} structures using {args.jobs} jobs...")

    worker = partial(_analyze_one, preset=args.preset)
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            results = p.map(worker, paths)
    else:
        results = [worker(p) for p in paths]

    results = [r for result in results if (r := result) is not None]

    if not results:
        print("No results to save.")
        return 1

    keys = results[0].keys()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved descriptors for {len(results)} structures to {args.out}")
    return 0


def _run_binding_site(args: argparse.Namespace) -> int:
    import csv

    try:
        ligand = _parse_ligand(args.ligand)
        mol = fetch(args.fetch) if args.fetch else read(args.file)
        site = mol.binding_site(ligand=ligand, cutoff=args.cutoff)
    except ValueError as e:
        print(f"Binding-site analysis failed: {e}", file=sys.stderr)
        return 2

    source = args.fetch if args.fetch else args.file
    rows = [
        {
            "file": source,
            "ligand_chain": site.ligand.chain,
            "ligand_resid": site.ligand.resid,
            "ligand_insertion_code": site.ligand.insertion_code,
            "ligand_resname": site.ligand.resname,
            "cutoff": site.cutoff,
            **record,
        }
        for record in site.to_records()
    ]
    _write_binding_site_csv(args.out, rows)

    if args.descriptors_out:
        desc = {"file": source, **site.descriptors(mol, preset="pocket-basic")}
        with open(args.descriptors_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(desc))
            writer.writeheader()
            writer.writerow(desc)

    print(f"Saved {len(rows)} binding-site residue records to {args.out}")
    return 0


def _write_binding_site_csv(path: str, rows: list[dict]) -> None:
    import csv

    fieldnames = [
        "file",
        "ligand_chain",
        "ligand_resid",
        "ligand_insertion_code",
        "ligand_resname",
        "cutoff",
        "residue_id",
        "chain",
        "resid",
        "insertion_code",
        "resname",
        "min_distance",
        "n_atom_contacts",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_one(path: str, to_fmt: str, out_dir: str, kwargs: dict) -> bool:
    """Export one structure's graph (worker; must be top-level so it is picklable
    under the ``spawn`` start method on macOS/Windows)."""
    try:
        mol = read(path)
        g = mol.to_graph()
        stem = os.path.splitext(os.path.basename(path))[0]

        if to_fmt == "pyg":
            import torch
            data = g.to_pyg_data(**kwargs)
            out_path = os.path.join(out_dir, f"{stem}.pt")
            torch.save(data, out_path)
        elif to_fmt == "dgl":
            from dgl.data.utils import save_graphs
            dg = g.to_dgl_graph(**kwargs)
            out_path = os.path.join(out_dir, f"{stem}.bin")
            save_graphs(out_path, [dg])
        elif to_fmt == "nx":
            import json

            import networkx as nx
            # NetworkX exporter doesn't support the new options yet
            ng = g.to_networkx()
            out_path = os.path.join(out_dir, f"{stem}.json")
            with open(out_path, "w") as f:
                json.dump(nx.node_link_data(ng), f)
        return True
    except Exception as e:
        print(f"Error exporting {path}: {e}", file=sys.stderr)
        return False


def _run_export(args: argparse.Namespace) -> int:
    from functools import partial
    from multiprocessing import Pool

    paths = _expand_globs(args.files)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Exporting {len(paths)} structures to {args.to} format...")

    kwargs = {
        "include_self_loops": args.self_loops,
        "include_global_node": args.global_node,
        "include_pe": args.pe,
        "pe_k": args.pe_k,
    }
    worker = partial(_export_one, to_fmt=args.to, out_dir=args.out_dir, kwargs=kwargs)
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            successes = p.map(worker, paths)
    else:
        successes = [worker(p) for p in paths]

    print(f"Successfully exported {sum(successes)} structures to {args.out_dir}")
    return 0



def _expand_globs(patterns: list[str]) -> list[str]:
    paths = []
    for p in patterns:
        if "*" in p or "?" in p:
            paths.extend(glob.glob(p, recursive=True))
        else:
            paths.append(p)
    return sorted(list(set(paths)))


if __name__ == "__main__":
    raise SystemExit(main())
