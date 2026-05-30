import os
import xml.etree.ElementTree as ET

import numpy as np

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


def test_write_cg_openmm_xml(tmp_path):
    cg = two_alanines().coarse_grain("martini")
    xml_path = str(tmp_path / "ALA.xml")
    
    # Write OpenMM XML
    ms.write_cg_openmm_xml(cg, xml_path)
    assert os.path.exists(xml_path)
    
    # Parse and verify XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    assert root.tag == "ForceField"
    residues = root.find("Residues")
    assert residues is not None
    
    res = residues.find("Residue")
    assert res is not None
    assert res.attrib["name"] == "ALA"
    
    atoms = res.findall("Atom")
    assert len(atoms) == 2
    assert {a.attrib["name"] for a in atoms} == {"BB", "SC"}
    assert {a.attrib["type"] for a in atoms} == {"CG_ALA_BB", "CG_ALA_SC"}
    assert {a.attrib["charge"] for a in atoms} == {"0.0"}
    
    bonds = res.findall("Bond")
    assert len(bonds) == 1
    assert bonds[0].attrib["atomName1"] == "BB"
    assert bonds[0].attrib["atomName2"] == "SC"
    
    external = res.findall("ExternalBond")
    assert len(external) == 1
    assert external[0].attrib["atomName"] == "BB"


def test_write_cg_openmm_xml_no_residues(tmp_path):
    # Dummy molecule without residue information
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = Molecule(coords, ["C", "C"])
    # Index mapping
    cg = mol.coarse_grain({"head": [0], "tail": [1]}, bonds=[(0, 1)])
    xml_path = str(tmp_path / "CG.xml")
    
    ms.write_cg_openmm_xml(cg, xml_path)
    assert os.path.exists(xml_path)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    res = root.find("Residues").find("Residue")
    assert res.attrib["name"] == "MOL"
    
    atoms = res.findall("Atom")
    assert len(atoms) == 2
    assert {a.attrib["name"] for a in atoms} == {"head", "tail"}
    
    bonds = res.findall("Bond")
    assert len(bonds) == 1
    assert bonds[0].attrib["atomName1"] == "head"
    assert bonds[0].attrib["atomName2"] == "tail"
