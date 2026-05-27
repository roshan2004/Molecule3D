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
    mine_pm = mine.principal_moments()
    ref_pm = np.sort(np.linalg.eigvalsh(u.atoms.moment_of_inertia()))
    assert np.allclose(mine_pm, ref_pm, rtol=1e-5)


def test_inertia_tensor_matches_mdanalysis(both):
    mine, u = both
    assert np.allclose(mine.inertia_tensor(), u.atoms.moment_of_inertia(), rtol=1e-5)


def test_centroid_matches_mdanalysis(both):
    mine, u = both
    assert np.allclose(mine.centroid, u.atoms.center_of_geometry(), atol=1e-5)


def test_distance_angle_dihedral_match_mdanalysis(both):
    from MDAnalysis.lib import distances as mdadist

    mine, u = both
    p = u.atoms.positions
    i, j, k, m = 0, 10, 20, 30
    assert mine.distance(i, j) == pytest.approx(float(mdadist.calc_bonds(p[i], p[j])), rel=1e-5)
    assert mine.angle(i, j, k) == pytest.approx(
        np.degrees(float(mdadist.calc_angles(p[i], p[j], p[k]))), abs=1e-4
    )
    assert mine.dihedral(i, j, k, m) == pytest.approx(
        np.degrees(float(mdadist.calc_dihedrals(p[i], p[j], p[k], p[m]))), abs=1e-4
    )


def test_rmsf_matches_mdanalysis(mda):
    from MDAnalysis.analysis import align, rms

    models = ms.read_pdb_models(ENSEMBLE)
    mine = ms.ensemble.rmsf(models)               # aligns each model to model 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(ENSEMBLE, in_memory=True)
        align.AlignTraj(u, u, select="all", ref_frame=0, in_memory=True).run()
        ref = rms.RMSF(u.atoms).run().results.rmsf
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-3)


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
