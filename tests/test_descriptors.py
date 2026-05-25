import numpy as np
import pytest

import molecule3d as m3d
from molecule3d import Molecule
from molecule3d.descriptors import flatten_descriptors, inertia_tensor


def water():
    return Molecule(
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        ["O", "H", "H"],
        name="water",
    )


def test_descriptors_include_scalar_and_vector_features():
    desc = water().descriptors(distance_bins=4, distance_range=(0.0, 4.0))
    assert desc["n_atoms"] == 3.0
    assert desc["count_H"] == 2.0
    assert desc["count_O"] == 1.0
    assert desc["radius_of_gyration"] > 0.0
    assert desc["bond_count"] == 2.0
    assert len(desc["inertia_tensor"]) == 9
    assert len(desc["principal_moments"]) == 3
    assert len(desc["principal_axes"]) == 9
    assert len(desc["distance_histogram"]) == 4
    assert sum(desc["distance_histogram"]) == 3.0


def test_inertia_tensor_is_symmetric_and_shape_anisotropy_finite():
    desc = water().descriptors()
    tensor = inertia_tensor(water())
    np.testing.assert_allclose(tensor, tensor.T)
    assert desc["shape_anisotropy"] >= 0.0
    assert desc["shape_anisotropy"] <= 1.0


def test_flatten_descriptors_expands_vector_features():
    flat = flatten_descriptors({"n_atoms": 3.0, "principal_moments": [1.0, 2.0, 3.0]})
    assert flat == {
        "n_atoms": 3.0,
        "principal_moments_0": 1.0,
        "principal_moments_1": 2.0,
        "principal_moments_2": 3.0,
    }


def test_featurize_many_returns_matrix_and_feature_names(tmp_path):
    water_path = tmp_path / "water.xyz"
    carbon_path = tmp_path / "carbon.xyz"
    m3d.write_xyz(water(), str(water_path))
    m3d.write_xyz(Molecule(np.array([[0.0, 0.0, 0.0]]), ["C"], name="carbon"), str(carbon_path))

    x, names = m3d.featurize_many(
        [str(water_path), str(carbon_path)],
        return_names=True,
        distance_bins=3,
        distance_range=(0.0, 3.0),
    )
    assert x.shape == (2, len(names))
    assert "n_atoms" in names
    assert "count_C" in names
    assert "distance_histogram_0" in names
    assert x[0, names.index("n_atoms")] == pytest.approx(3.0)
    assert x[1, names.index("count_C")] == pytest.approx(1.0)


def test_featurize_many_accepts_explicit_feature_names(tmp_path):
    path = tmp_path / "water.xyz"
    m3d.write_xyz(water(), str(path))
    x = m3d.featurize_many([str(path)], feature_names=["n_atoms", "count_O"])
    np.testing.assert_allclose(x, [[3.0, 1.0]])
