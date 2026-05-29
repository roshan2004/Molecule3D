"""Tests for the optional MCP server (``molscope.mcp_server``).

The whole module is skipped when the ``mcp`` extra is not installed (for example
on Python 3.9, where the reference SDK is unavailable). Tools are exercised
through the real FastMCP ``call_tool`` surface so the registered schemas and the
adapter logic are both covered. Every case uses bundled local structures, so no
network access is needed.
"""

import asyncio
import json
import os

import pytest

pytest.importorskip("mcp")

from molscope.mcp_server import _load, build_server  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
UBQ = os.path.join(DATA, "1ubq.pdb")
TRYPSIN = os.path.join(DATA, "3ptb.pdb")


@pytest.fixture(scope="module")
def server():
    return build_server()


def _content(server, name, args):
    """Return the content blocks from a tool call.

    Text/JSON tools return ``(content, structured)``; image tools (no structured
    output) return the bare content list. Handle both shapes.
    """
    result = asyncio.run(server.call_tool(name, args))
    return result[0] if isinstance(result, tuple) else result


def _text(server, name, **args):
    return _content(server, name, args)[0].text


def _json(server, name, **args):
    return json.loads(_text(server, name, **args))


def _image(server, name, **args):
    return _content(server, name, args)[0]


EXPECTED_TOOLS = {
    "summarize_structure",
    "compute_descriptors",
    "secondary_structure",
    "contact_map",
    "binding_site",
    "molecular_graph",
    "coarse_grain",
    "render_structure",
    "render_contact_map",
}


def test_server_registers_expected_tools(server):
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == EXPECTED_TOOLS


def test_summarize_structure(server):
    out = _text(server, "summarize_structure", source=UBQ)
    assert "atoms" in out and "chains A" in out


def test_compute_descriptors_batch(server):
    out = _json(server, "compute_descriptors", sources=[UBQ, TRYPSIN])
    assert out["n_features"] == len(out["feature_names"]) > 0
    assert [row["source"] for row in out["rows"]] == [UBQ, TRYPSIN]
    # Each row carries one value per named feature.
    for row in out["rows"]:
        assert len(row["values"]) == out["n_features"]


def test_secondary_structure(server):
    out = _json(server, "secondary_structure", source=UBQ)
    comp = out["composition"]
    assert out["n_residues"] == 76
    assert comp["helix"] + comp["strand"] + comp["coil"] == out["n_residues"]
    assert {"chain", "resid", "resname", "code"} <= set(out["residues"][0])


def test_contact_map(server):
    out = _json(server, "contact_map", source=UBQ, cutoff=8.0)
    assert out["level"] == "residue"
    assert out["n_contacts"] > 0
    assert out["n_contacts"] == len(out["pairs"]) or out["pairs_truncated"]


def test_binding_site_auto_ligand(server):
    out = _json(server, "binding_site", source=TRYPSIN)
    assert out["n_residues"] > 0
    first = out["residues"][0]
    assert first["min_distance"] <= out["cutoff"]


def test_molecular_graph(server):
    out = _json(server, "molecular_graph", source=UBQ)
    assert out["n_nodes"] == 660
    assert out["n_edges"] > 0
    assert out["node_feature_matrix_shape"][0] == out["n_nodes"]
    assert out["node_feature_names"] and out["edge_feature_names"]


def test_coarse_grain(server):
    out = _json(server, "coarse_grain", source=UBQ, mapping="residue_com")
    assert out["mapping"] == "residue_com"
    # One bead per residue: 76 protein residues plus crystallographic waters.
    assert out["n_beads"] >= 76
    assert out["n_dropped_atoms"] == 0
    assert out["n_bonds"] == out["n_beads"] - 1


def test_render_structure_returns_png(server):
    block = _image(server, "render_structure", source=UBQ, color_by="chain")
    assert getattr(block, "mimeType", None) == "image/png"
    assert block.data  # base64 payload present


def test_render_contact_map_returns_png(server):
    block = _image(server, "render_contact_map", source=UBQ)
    assert getattr(block, "mimeType", None) == "image/png"
    assert block.data


def test_load_accepts_paths_and_rejects_garbage():
    assert _load(UBQ).summary()
    with pytest.raises(FileNotFoundError):
        _load("not-a-file-or-id.zzz")


def test_load_dispatches_pdb_id_to_fetch(monkeypatch):
    import molscope.io as mio

    sentinel = object()
    monkeypatch.setattr(mio, "fetch", lambda pdb_id: sentinel)
    assert _load("1abc") is sentinel


def test_jsonable_coerces_numpy():
    import numpy as np

    from molscope.mcp_server import _jsonable

    assert _jsonable(np.float64(3.5)) == 3.5
    assert _jsonable(np.array([1, 2])) == [1, 2]
    assert _jsonable("x") == "x"


def test_main_without_mcp_raises_systemexit(monkeypatch):
    import molscope.mcp_server as srv

    def boom():
        raise ImportError("mcp not installed")

    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit):
        srv.main()


def test_main_runs_server(monkeypatch):
    import molscope.mcp_server as srv

    calls = {}

    class _FakeServer:
        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(srv, "build_server", lambda: _FakeServer())
    srv.main()
    assert calls.get("ran") is True
