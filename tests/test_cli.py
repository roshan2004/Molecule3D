"""Tests for CLI argument helpers."""

import pytest

from molscope.cli import _default_to_view, _parse_ligand, _parse_selection


def test_default_to_view_keeps_subcommands_and_top_level_help():
    subcommands = {"view", "analyze", "binding-site", "export"}

    assert _default_to_view(["analyze", "a.pdb"], subcommands) == ["analyze", "a.pdb"]
    assert _default_to_view(["binding-site", "a.pdb"], subcommands) == [
        "binding-site",
        "a.pdb",
    ]
    assert _default_to_view(["--help"], subcommands) == ["--help"]


def test_default_to_view_accepts_leading_view_options():
    subcommands = {"view", "analyze", "export"}

    assert _default_to_view(["--fetch", "1aml"], subcommands) == ["view", "--fetch", "1aml"]


def test_parse_selection_accepts_single_key_value():
    assert _parse_selection("atom_name=CA") == {"atom_name": "CA"}


def test_parse_selection_accepts_and_expression():
    assert _parse_selection("chain=A and atom_name=CA") == {
        "chain": "A",
        "atom_name": "CA",
    }


def test_parse_selection_accepts_repeated_flags():
    assert _parse_selection(["chain=A", "atom_name=CA"]) == {
        "chain": "A",
        "atom_name": "CA",
    }


def test_parse_selection_coerces_resid_and_hetero_values():
    assert _parse_selection("resid=10-20 and hetero=false") == {
        "resid": (10, 20),
        "hetero": False,
    }


def test_parse_selection_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unsupported field"):
        _parse_selection("name=CA")


def test_parse_ligand_accepts_resname_and_location():
    assert _parse_ligand(None) is None
    assert _parse_ligand("BEN") == "BEN"
    assert _parse_ligand("A:1") == ("A", 1)


def test_parse_ligand_rejects_bad_location():
    with pytest.raises(ValueError, match="integer resid"):
        _parse_ligand("A:BEN")
