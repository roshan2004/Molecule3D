"""Tests for CLI argument helpers."""

import pytest

from molscope.cli import _default_to_view, _parse_selection


def test_default_to_view_keeps_subcommands_and_top_level_help():
    subcommands = {"view", "analyze", "export"}

    assert _default_to_view(["analyze", "a.pdb"], subcommands) == ["analyze", "a.pdb"]
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
