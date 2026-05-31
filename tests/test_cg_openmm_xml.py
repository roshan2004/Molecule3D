import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule


def two_alanines(second_chain="A"):
    names = ["N", "CA", "C", "O", "CB"] * 2
    els = ["N", "C", "C", "O", "C"] * 2
    return Molecule(
        np.arange(30).reshape(10, 3).astype(float),
        els,
        name="dialanine",
        atom_names=names,
        resnames=["ALA"] * 10,
        resids=np.array([1] * 5 + [2] * 5),
        chains=["A"] * 5 + [second_chain] * 5,
    )


def _parse(xml_path):
    """Return (atom_types, residues) parsed from a written ForceField XML.

    ``atom_types`` is the set of ``<Type name=...>`` entries; ``residues`` maps
    each residue name to a dict with its bead names, internal bond name-pairs,
    external bond names, and the types each bead references.
    """
    root = ET.parse(xml_path).getroot()
    assert root.tag == "ForceField"

    atom_types = {t.attrib["name"] for t in root.find("AtomTypes").findall("Type")}

    residues = {}
    for res in root.find("Residues").findall("Residue"):
        atoms = res.findall("Atom")
        residues[res.attrib["name"]] = {
            "beads": {a.attrib["name"] for a in atoms},
            "types": {a.attrib["name"]: a.attrib["type"] for a in atoms},
            "bonds": {
                tuple(sorted((b.attrib["atomName1"], b.attrib["atomName2"])))
                for b in res.findall("Bond")
            },
            "external": {e.attrib["atomName"] for e in res.findall("ExternalBond")},
        }
    return atom_types, residues


def test_write_cg_openmm_xml_structure(tmp_path):
    cg = two_alanines().coarse_grain("martini")
    xml_path = str(tmp_path / "ALA.xml")
    ms.write_cg_openmm_xml(cg, xml_path)
    assert os.path.exists(xml_path)

    atom_types, residues = _parse(xml_path)

    assert set(residues) == {"ALA"}
    ala = residues["ALA"]
    assert ala["beads"] == {"BB", "SC"}
    assert ala["types"] == {"BB": "CG_ALA_BB", "SC": "CG_ALA_SC"}
    assert ala["bonds"] == {("BB", "SC")}
    assert ala["external"] == {"BB"}

    # Every type a residue atom references must be defined in <AtomTypes>.
    assert atom_types == {"CG_ALA_BB", "CG_ALA_SC"}


def test_write_cg_openmm_xml_no_residues(tmp_path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = Molecule(coords, ["C", "C"])
    cg = mol.coarse_grain({"head": [0], "tail": [1]}, bonds=[(0, 1)])
    xml_path = str(tmp_path / "CG.xml")
    ms.write_cg_openmm_xml(cg, xml_path)

    atom_types, residues = _parse(xml_path)
    assert set(residues) == {"MOL"}
    assert residues["MOL"]["beads"] == {"head", "tail"}
    assert residues["MOL"]["bonds"] == {("head", "tail")}
    assert atom_types == {"CG_MOL_head", "CG_MOL_tail"}


def test_write_cg_openmm_xml_roundtrip_is_self_consistent(tmp_path):
    """The written file must be internally consistent (no dangling references)."""
    cg = two_alanines().coarse_grain("martini")
    xml_path = str(tmp_path / "ALA.xml")
    ms.write_cg_openmm_xml(cg, xml_path)

    atom_types, residues = _parse(xml_path)

    # Residue names in the file == unique residue names in the CG model.
    assert set(residues) == set(cg.resnames)

    for res in residues.values():
        # Every referenced type is defined exactly once in <AtomTypes>.
        for type_name in res["types"].values():
            assert type_name in atom_types
        # Every bond / external-bond atom names an actual bead in the residue.
        for a, b in res["bonds"]:
            assert a in res["beads"] and b in res["beads"]
        for atom in res["external"]:
            assert atom in res["beads"]


def test_write_cg_openmm_xml_atom_types_carry_mass(tmp_path):
    cg = two_alanines().coarse_grain("martini")
    xml_path = str(tmp_path / "ALA.xml")
    ms.write_cg_openmm_xml(cg, xml_path)

    root = ET.parse(xml_path).getroot()
    types = root.find("AtomTypes").findall("Type")
    assert types, "expected at least one <Type> entry"
    for t in types:
        # A CG bead has no element, so OpenMM requires an explicit mass.
        assert "element" not in t.attrib
        assert float(t.attrib["mass"]) > 0


def test_write_cg_openmm_xml_loads_in_openmm(tmp_path):
    """Gold-standard check: OpenMM accepts the file and matches it to a topology."""
    openmm_app = pytest.importorskip("openmm.app", reason="openmm not installed")

    cg = two_alanines().coarse_grain("martini")
    xml_path = str(tmp_path / "ALA.xml")
    ms.write_cg_openmm_xml(cg, xml_path)

    # Loads as a standalone ForceField (this used to raise KeyError because the
    # referenced atom types were never defined).
    ff = openmm_app.ForceField(xml_path)

    # Build a topology of two ALA beads (BB-SC internally, BB-BB backbone) and
    # confirm OpenMM matches our residue template to each residue by graph
    # isomorphism on atoms + bonds + external bonds.
    top = openmm_app.Topology()
    chain = top.addChain()
    backbone = []
    for _ in range(2):
        res = top.addResidue("ALA", chain)
        bb = top.addAtom("BB", None, res)
        sc = top.addAtom("SC", None, res)
        top.addBond(bb, sc)
        backbone.append(bb)
    top.addBond(backbone[0], backbone[1])

    matched = ff.getMatchingTemplates(top)
    assert [t.name for t in matched] == ["ALA", "ALA"]
