"""Command-line entry point for MolScope.

Supports viewing single structures, batch analysis to CSV, and batch graph export.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from .io import fetch, read


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
        "--select", metavar="SPEC",
        help="atom selection, e.g. 'chain=A' or 'atom_name=CA'",
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

    # Default to 'view' if no subcommand provided
    if argv is None:
        argv = sys.argv[1:]
    if not argv or (argv[0] not in subparsers.choices and not argv[0].startswith("-")):
        argv = ["view"] + argv

    args = parser.parse_args(argv)

    if args.command == "view":
        return _run_view(args)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "export":
        return _run_export(args)

    return 0


def _run_view(args: argparse.Namespace) -> int:
    mol = fetch(args.fetch) if args.fetch else read(args.file)
    if args.select:
        key, _, value = args.select.partition("=")
        mol = mol.select(**{key.strip(): value.strip()})
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
