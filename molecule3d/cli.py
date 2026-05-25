"""Command-line entry point: ``python -m molecule3d FILE [options]``."""

from __future__ import annotations

import argparse

from .io import read


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="molecule3d",
        description="Read a .xyz or .pdb file and plot its atoms in 3D.",
    )
    parser.add_argument("file", help="path to a .xyz or .pdb structure file")
    parser.add_argument(
        "--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
        help="shift all atoms by this vector before plotting",
    )
    parser.add_argument(
        "--center", action="store_true", help="move the centroid to the origin",
    )
    parser.add_argument(
        "--rotate", nargs=2, metavar=("AXIS", "DEG"),
        help="rotate about AXIS (x/y/z) by DEG degrees, e.g. --rotate z 90",
    )
    bonds = parser.add_mutually_exclusive_group()
    bonds.add_argument("--bonds", dest="bonds", action="store_true",
                       help="force drawing inferred bonds")
    bonds.add_argument("--no-bonds", dest="bonds", action="store_false",
                       help="never draw bonds")
    parser.set_defaults(bonds=None)
    parser.add_argument("--save", metavar="PATH", help="save the figure instead of showing it")

    args = parser.parse_args(argv)

    mol = read(args.file)
    if args.center:
        mol = mol.centered()
    if args.translate:
        mol = mol.translate(args.translate)
    if args.rotate:
        mol = mol.rotate(args.rotate[0], float(args.rotate[1]))

    print(f"{mol.name}: {len(mol)} atoms")

    show = args.save is None
    ax = mol.plot(show_bonds=args.bonds, show=show)
    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
