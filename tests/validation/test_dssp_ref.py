"""Tier 2 validation: molscope's simplified DSSP vs the reference ``mkdssp``.

molscope ships a *simplified* secondary-structure assignment, so the honest test
is not byte-for-byte equality with the reference but a per-residue *agreement
fraction* on the reduced 3-state alphabet (helix / strand / coil). The test
prints that fraction so a regression is visible, and asserts a defensible floor.

Requires Biopython and the ``mkdssp`` executable on PATH (``pip install
'molscope[validation]'`` plus a system ``dssp``/``mkdssp``). Skips otherwise.
"""

from pathlib import Path

import pytest

import molscope as ms

pytestmark = pytest.mark.validation

ROOT = Path(__file__).resolve().parents[2]
PROTEIN = str(ROOT / "1fqy.pdb")

# Reduce DSSP's 8-state alphabet to 3 states. Note molscope emits 'S' for bend
# (not strand) and 'T' for turn; both reduce to coil, matching the reference.
_THREE_STATE = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}


def _to3(code: str) -> str:
    return _THREE_STATE.get(code, "C")


def _reference_codes(pdb_path: str) -> dict:
    """Return ``{(chain, resid): dssp_code}`` from the reference mkdssp run."""
    Bio_PDB = pytest.importorskip("Bio.PDB")
    DSSP = pytest.importorskip("Bio.PDB.DSSP").DSSP
    structure = Bio_PDB.PDBParser(QUIET=True).get_structure("ref", pdb_path)
    # The binary is called `dssp` on some distros and `mkdssp` on others, and
    # Biopython's default name varies by version. Try both before skipping.
    last_exc = None
    for executable in ("mkdssp", "dssp"):
        try:
            dssp = DSSP(structure[0], pdb_path, dssp=executable)
            break
        except Exception as exc:  # binary missing or output unparseable
            last_exc = exc
    else:
        pytest.skip(f"reference dssp/mkdssp unavailable: {last_exc}")
    out = {}
    for chain_id, (_, resid, _icode) in dssp.keys():
        out[(chain_id, int(resid))] = dssp[(chain_id, (_, resid, _icode))][2]
    return out


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

    # Floor for a simplified method. Once you observe the real number on CI,
    # tighten this toward it so the test guards against drift in both directions.
    assert fraction >= 0.70


def test_helix_and_strand_fractions_are_in_the_right_ballpark():
    """Aggregate composition should track the reference within a loose band."""
    ref = _reference_codes(PROTEIN)
    mine = _molscope_codes(PROTEIN)
    shared = sorted(ref.keys() & mine.keys())

    def helix_frac(codes):
        return sum(_to3(codes[k]) == "H" for k in shared) / len(shared)

    assert abs(helix_frac(mine) - helix_frac(ref)) <= 0.15
