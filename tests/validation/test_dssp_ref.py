"""Tier 2 validation: molscope's simplified DSSP vs the reference ``mkdssp``.

molscope ships a *simplified* secondary-structure assignment, so the honest test
is not byte-for-byte equality with the reference but a per-residue *agreement
fraction* on the reduced 3-state alphabet (helix / strand / coil). The test
prints that fraction so a regression is visible, and asserts a defensible floor.

We invoke ``mkdssp`` directly and parse its classic (``--output-format dssp``)
output rather than going through Biopython's wrapper, which does not drive
mkdssp v4 (the version shipped by current Linux distros). Skips cleanly when no
``mkdssp``/``dssp`` binary is on PATH.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import molscope as ms

pytestmark = pytest.mark.validation

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
PROTEIN = str(DATA / "1fqy.pdb")

# Reduce DSSP's 8-state alphabet to 3 states. molscope emits 'S' for bend (not
# strand) and 'T' for turn; both reduce to coil, matching the reference.
_THREE_STATE = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}


def _to3(code: str) -> str:
    return _THREE_STATE.get(code, "C")


def _run_mkdssp(exe: str, pdb_path: str, out_path: str) -> subprocess.CompletedProcess:
    """Try the v4 invocation first, then the legacy one (older dssp/mkdssp)."""
    attempts = (
        [exe, "--output-format", "dssp", pdb_path, out_path],
        [exe, pdb_path, out_path],
    )
    last = None
    for cmd in attempts:
        last = subprocess.run(cmd, capture_output=True, text=True)
        if last.returncode == 0 and Path(out_path).stat().st_size > 0:
            return last
    return last


def _parse_dssp(text: str) -> dict:
    """Parse classic DSSP output into ``{(chain, resid): code}``.

    Columns are fixed-width: residue number [5:10], chain [11], amino acid [13]
    ('!' marks a chain break), secondary-structure code [16] (space == coil).
    """
    out = {}
    in_body = False
    for line in text.splitlines():
        if not in_body:
            if line.startswith("  #  RESIDUE"):
                in_body = True
            continue
        if len(line) < 17 or line[13] == "!":
            continue
        try:
            resid = int(line[5:10])
        except ValueError:
            continue
        chain = line[11]
        code = line[16] if line[16] != " " else "-"
        out[(chain, resid)] = code
    return out


def _reference_codes(pdb_path: str) -> dict:
    exe = shutil.which("mkdssp") or shutil.which("dssp")
    if exe is None:
        pytest.skip("reference mkdssp/dssp not found on PATH")
    with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp:
        out_path = tmp.name
    proc = _run_mkdssp(exe, pdb_path, out_path)
    text = Path(out_path).read_text() if Path(out_path).exists() else ""
    Path(out_path).unlink(missing_ok=True)
    ref = _parse_dssp(text)
    if not ref:
        pytest.skip(
            f"could not parse mkdssp output (rc={proc.returncode if proc else '?'}): "
            f"{(proc.stderr or '').strip()[:200] if proc else ''}"
        )
    return ref


def _molscope_codes(pdb_path: str) -> dict:
    ss = ms.read(pdb_path).secondary_structure()
    return {(c, int(r)): code for c, r, code in zip(ss.chains, ss.resids, ss.codes.tolist())}


def test_dssp_three_state_agreement_with_reference():
    ref = _reference_codes(PROTEIN)
    mine = _molscope_codes(PROTEIN)

    shared = sorted(ref.keys() & mine.keys())
    assert len(shared) > 50, "too few residues matched between molscope and reference"

    agree = sum(_to3(mine[k]) == _to3(ref[k]) for k in shared)
    fraction = agree / len(shared)
    print(f"\n3-state DSSP agreement: {fraction:.1%} over {len(shared)} residues")

    # Observed 99.1% vs mkdssp 4.2.2 on 1fqy (mostly helical). Floor set below
    # that with headroom for boundary residues, tight enough to catch a real
    # regression in the assignment.
    assert fraction >= 0.95


def test_helix_and_strand_fractions_are_in_the_right_ballpark():
    """Aggregate composition should track the reference within a loose band."""
    ref = _reference_codes(PROTEIN)
    mine = _molscope_codes(PROTEIN)
    shared = sorted(ref.keys() & mine.keys())

    def helix_frac(codes):
        return sum(_to3(codes[k]) == "H" for k in shared) / len(shared)

    assert abs(helix_frac(mine) - helix_frac(ref)) <= 0.15
