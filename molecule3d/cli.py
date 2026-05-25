"""Command-line entry point: ``python -m molecule3d FILE [options]``."""

from __future__ import annotations

import argparse

from .io import fetch, read


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="molecule3d",
        description="Read a structure (.xyz/.pdb/.cif/.sdf), transform it, and plot in 3D.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="path to a structure file")
    src.add_argument("--fetch", metavar="PDBID", help="download a structure from RCSB by id")

    parser.add_argument(
        "--select", metavar="SPEC",
        help="atom selection, e.g. 'chain=A' or 'atom_name=CA' or 'element=C'",
    )
    parser.add_argument(
        "--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
        help="shift all atoms by this vector before plotting",
    )
    parser.add_argument("--center", action="store_true", help="move the centroid to the origin")
    parser.add_argument(
        "--rotate", nargs=2, metavar=("AXIS", "DEG"),
        help="rotate about AXIS (x/y/z) by DEG degrees, e.g. --rotate z 90",
    )
    parser.add_argument(
        "--color-by", choices=["element", "chain", "residue"], default="element",
        help="how to colour atoms (default: element)",
    )
    bonds = parser.add_mutually_exclusive_group()
    bonds.add_argument("--bonds", dest="bonds", action="store_true", help="force drawing bonds")
    bonds.add_argument("--no-bonds", dest="bonds", action="store_false", help="never draw bonds")
    parser.set_defaults(bonds=None)
    parser.add_argument("--save", metavar="PATH", help="save the figure instead of showing it")
    parser.add_argument("--gif", metavar="PATH", help="save a spinning animation as a GIF")

    args = parser.parse_args(argv)

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
    ax = mol.plot(show_bonds=args.bonds, color_by=args.color_by, show=show)
    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
