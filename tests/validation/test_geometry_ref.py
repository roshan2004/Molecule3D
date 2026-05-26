"""Tier 2 validation: geometry and RMSD vs MDAnalysis.

These compare molscope's mass-weighted geometry (radius of gyration, centre of
mass, principal moments of inertia) and its Kabsch RMSD against MDAnalysis on
the bundled structures. Both libraries agree to floating-point noise in
practice, so the tolerances here are tight on purpose: a loose tolerance would
not be a test. Skips when MDAnalysis is not installed.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import molscope as ms

pytestmark = pytest.mark.validation

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
PROTEIN = str(DATA / "1fqy.pdb")
ENSEMBLE = str(DATA / "1aml.pdb")


@pytest.fixture(scope="module")
def mda():
    return pytest.importorskip("MDAnalysis")


@pytest.fixture(scope="module")
def both(mda):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MDAnalysis mass/element guessing chatter
        return ms.read(PROTEIN), mda.Universe(PROTEIN)


def test_radius_of_gyration_matches_mdanalysis(both):
    mine, u = both
    assert mine.radius_of_gyration == pytest.approx(u.atoms.radius_of_gyration(), rel=1e-6)


def test_center_of_mass_matches_mdanalysis(both):
    mine, u = both
    assert np.allclose(mine.center_of_mass, u.atoms.center_of_mass(), atol=1e-5)


def test_principal_moments_match_mdanalysis(both):
    mine, u = both
    mine_pm = np.array(mine.descriptors()["principal_moments"])
    ref_pm = np.sort(np.linalg.eigvalsh(u.atoms.moment_of_inertia()))
    assert np.allclose(mine_pm, ref_pm, rtol=1e-5)


def test_kabsch_rmsd_matches_mdanalysis(mda):
    from MDAnalysis.analysis import rms

    m1 = ms.read_pdb(ENSEMBLE, model=1)
    m2 = ms.read_pdb(ENSEMBLE, model=2)
    mine = m1.rmsd(m2, align=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(ENSEMBLE)
        u.trajectory[0]
        p0 = u.atoms.positions.copy()
        u.trajectory[1]
        p1 = u.atoms.positions.copy()
    assert len(m1) == len(u.atoms)  # same atom selection/order
    assert mine == pytest.approx(rms.rmsd(p0, p1, superposition=True), abs=1e-4)
