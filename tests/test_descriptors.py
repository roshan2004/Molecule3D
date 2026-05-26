import numpy as np
import pytest

import molscope as ms
from molscope import Molecule
from molscope.descriptors import (
    RDKIT_BASIC_DESCRIPTORS,
    _pairwise_distance_histogram,
    _pairwise_distances,
    descriptor_feature_names,
    flatten_descriptors,
    inertia_tensor,
)


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
    assert desc["atom_contact_count"] == 3.0


def test_chunked_distance_histogram_matches_dense_result():
    coords = water().coords
    expected, _ = np.histogram(_pairwise_distances(coords), bins=4, range=(0.0, 4.0))
    hist = _pairwise_distance_histogram(
        coords,
        bins=4,
        distance_range=(0.0, 4.0),
        chunk_size=1,
    )
    np.testing.assert_allclose(hist, expected.astype(float))


def test_descriptors_validate_distance_chunk_size():
    with pytest.raises(ValueError, match="distance_chunk_size"):
        water().descriptors(distance_chunk_size=0)


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
    ms.write_xyz(water(), str(water_path))
    ms.write_xyz(Molecule(np.array([[0.0, 0.0, 0.0]]), ["C"], name="carbon"), str(carbon_path))

    x, names = ms.featurize_many(
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
    ms.write_xyz(water(), str(path))
    x = ms.featurize_many([str(path)], feature_names=["n_atoms", "count_O"])
    np.testing.assert_allclose(x, [[3.0, 1.0]])


def test_descriptor_presets_return_stable_feature_sets():
    desc = water().descriptors(preset="native-basic")
    assert "n_atoms" in desc
    assert "count_O" in desc
    assert "inertia_tensor" not in desc
    assert "distance_histogram" not in desc
    names = descriptor_feature_names("native-basic")
    assert "n_atoms" in names
    assert "distance_histogram_0" not in names


def test_native_3d_descriptor_preset_includes_flattened_vector_names():
    desc = water().descriptors(preset="native-3d", distance_bins=4, distance_range=(0.0, 4.0))
    flat = flatten_descriptors(desc)
    names = descriptor_feature_names("native-3d", distance_bins=4)
    assert "inertia_tensor" in desc
    assert "distance_histogram" in desc
    assert "distance_histogram_3" in names
    assert set(flat) == set(names)


def test_featurize_many_uses_preset_feature_order(tmp_path):
    water_path = tmp_path / "water.xyz"
    ms.write_xyz(water(), str(water_path))
    x, names = ms.featurize_many([str(water_path)], preset="native-basic", return_names=True)
    assert names == descriptor_feature_names("native-basic")
    assert x.shape == (1, len(names))


def test_unknown_descriptor_preset_raises():
    with pytest.raises(ValueError, match="unknown descriptor preset"):
        water().descriptors(preset="unknown")


def test_descriptors_can_include_selected_rdkit_descriptors():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["C", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    desc = mol.descriptors(include_rdkit=True, rdkit_descriptor_names=["MolWt"])
    assert desc["n_atoms"] == 2.0
    assert desc["rdkit_MolWt"] > 0.0


def test_rdkit_basic_descriptor_preset_is_stable():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["C", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    desc = mol.descriptors(preset="rdkit-basic")
    for name in RDKIT_BASIC_DESCRIPTORS:
        assert f"rdkit_{name}" in desc
    assert set(flatten_descriptors(desc)) == set(descriptor_feature_names("rdkit-basic"))
