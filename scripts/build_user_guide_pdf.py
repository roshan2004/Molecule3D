"""Build a PDF version of the MkDocs user guide with Pandoc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "docs" / "_build" / "molscope-user-guide.pdf"
USER_GUIDE_FILES = [
    ROOT / "docs" / "user-guide" / "reading-files.md",
    ROOT / "docs" / "user-guide" / "selections.md",
    ROOT / "docs" / "user-guide" / "geometry.md",
    ROOT / "docs" / "user-guide" / "plotting.md",
    ROOT / "docs" / "user-guide" / "ensembles.md",
    ROOT / "docs" / "user-guide" / "molecular-graphs.md",
    ROOT / "docs" / "user-guide" / "descriptors.md",
    ROOT / "docs" / "user-guide" / "coarse-graining.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"PDF path to write (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    _require("pandoc")
    engine = _pdf_engine()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "pandoc",
        *[str(path) for path in USER_GUIDE_FILES],
        "--standalone",
        "--toc",
        "--number-sections",
        "--pdf-engine",
        engine,
        "--metadata",
        "title=MolScope User Guide",
        "--metadata",
        "author=MolScope",
        "--metadata",
        "geometry=margin=1in",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print(output)
    return 0


def _require(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"{command!r} is required to build the PDF")


def _pdf_engine() -> str:
    for engine in ("xelatex", "pdflatex"):
        if shutil.which(engine):
            return engine
    raise SystemExit("a LaTeX PDF engine is required; install xelatex or pdflatex")


if __name__ == "__main__":
    raise SystemExit(main())
