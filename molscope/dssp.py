"""Simplified DSSP secondary-structure assignment, in pure NumPy.

This implements the Kabsch & Sander (1983) approach: backbone amide hydrogens
are placed geometrically, an electrostatic hydrogen-bond energy is computed
between every backbone C=O and N-H pair, and secondary structure is assigned
from the resulting hydrogen-bond pattern (helices from n-turns, strands from
bridges/ladders, turns and bends).

It is an **educational/prototyping** implementation: it covers the main DSSP
classes (H/G/I helices, E/B strands, T turns, S bends) but is not bit-identical
to the reference ``mkdssp`` program on every edge case. It needs backbone N, CA,
C and O atoms, so it works on proteins read from PDB/mmCIF, not on bare ``.xyz``.

Codes: ``H`` alpha-helix, ``G`` 3-10 helix, ``I`` pi-helix, ``E`` beta-strand,
``B`` beta-bridge, ``T`` turn, ``S`` bend, ``-`` coil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .molecule import Molecule

# Kabsch-Sander hydrogen-bond energy constants (gives kcal/mol with r in angstrom).
_Q1 = 0.42
_Q2 = 0.20
_F = 332.0
_HBOND_CUTOFF = -0.5      # energy below this counts as a hydrogen bond
_CA_CUTOFF = 9.0          # no H-bond if CA-CA further than this (angstrom)
_CHAIN_BREAK = 2.5        # C(i)-N(i+1) above this is a chain break (angstrom)

# Single-letter codes, highest assignment priority first.
_PRIORITY = ["H", "E", "B", "G", "I", "T", "S"]

#: Colours for ``Molecule.plot(color_by="ss")``.
SS_COLORS = {
    "H": "#e6194b",   # alpha-helix
    "G": "#f58231",   # 3-10 helix
    "I": "#911eb4",   # pi-helix
    "E": "#ffe119",   # beta-strand
    "B": "#bfef45",   # beta-bridge
    "T": "#42d4f4",   # turn
    "S": "#aaffc3",   # bend
    "-": "#d9d9d9",   # coil
}


# 8-state -> 3-state (helix / strand / coil) reduction.
_THREE_STATE = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}


def _ss_counts(codes: np.ndarray) -> dict:
    """Helix/strand/coil counts and fractions for a sequence of DSSP codes."""
    helix = int(np.isin(codes, ["H", "G", "I"]).sum())
    strand = int(np.isin(codes, ["E", "B"]).sum())
    total = len(codes)
    coil = total - helix - strand
    denom = total or 1
    return {
        "residues": total,
        "helix": helix, "strand": strand, "coil": coil,
        "helix_fraction": helix / denom,
        "strand_fraction": strand / denom,
        "coil_fraction": coil / denom,
    }


@dataclass(frozen=True)
class SSSegment:
    """A contiguous run of one DSSP code within a single chain."""

    code: str          # the DSSP code shared by the run
    chain: str         # chain id
    start: int         # first residue id (inclusive)
    end: int           # last residue id (inclusive)
    length: int        # number of residues in the run

    def __repr__(self) -> str:
        loc = f"{self.chain}:" if self.chain else ""
        return f"SSSegment({self.code} {loc}{self.start}-{self.end}, len={self.length})"


@dataclass(frozen=True)
class SecondaryStructure:
    """Per-residue DSSP assignment for a structure.

    ``codes`` holds one single-character code per residue, aligned with
    ``resids``/``chains``/``resnames`` (in chain/residue order).
    """

    codes: np.ndarray          # (R,) '<U1' DSSP codes
    resids: np.ndarray         # (R,) residue ids
    chains: list               # (R,) chain ids
    resnames: list             # (R,) residue names

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def string(self) -> str:
        """The assignment as a single string, e.g. ``'--HHHHH--EEEE--'``."""
        return "".join(self.codes.tolist())

    def simplified(self) -> str:
        """The assignment reduced to 3 states: ``H`` helix, ``E`` strand, ``C`` coil."""
        return "".join(_THREE_STATE.get(c, "C") for c in self.codes.tolist())

    def summary(self) -> dict:
        """Counts and fractions for helix, strand, and coil/other."""
        return _ss_counts(self.codes)

    def per_chain(self) -> dict:
        """``{chain: summary-dict}`` with the :meth:`summary` breakdown per chain."""
        out: dict[str, dict] = {}
        chains = np.asarray(self.chains)
        for chain in dict.fromkeys(self.chains):  # preserve first-seen order
            out[chain] = _ss_counts(self.codes[chains == chain])
        return out

    def segments(self, include_coil: bool = False) -> list[SSSegment]:
        """Contiguous runs of one code within a chain, as :class:`SSSegment` list.

        Coil (``'-'``) runs are omitted unless ``include_coil`` is true.
        """
        segments: list[SSSegment] = []
        n = len(self.codes)
        if n == 0:
            return segments
        codes = self.codes.tolist()
        start = 0
        for i in range(1, n + 1):
            split = i == n or codes[i] != codes[i - 1] or self.chains[i] != self.chains[i - 1]
            if split:
                code = codes[start]
                if include_coil or code != "-":
                    segments.append(SSSegment(
                        code=code, chain=self.chains[start],
                        start=int(self.resids[start]), end=int(self.resids[i - 1]),
                        length=i - start,
                    ))
                start = i
        return segments


def _backbone_residues(molecule: Molecule):
    """Extract per-residue N/CA/C/O coordinates for residues with a full backbone."""
    if not molecule.atom_names or len(molecule.resids) == 0:
        raise ValueError(
            "secondary-structure assignment needs per-atom names and residue ids "
            "(read the structure from PDB or mmCIF)"
        )
    names = molecule.atom_names
    coords = molecule.coords
    N, CA, C, O = [], [], [], []
    resids, chains, resnames = [], [], []
    for idx, resname, resid, chain in molecule.residue_groups():
        atoms = {names[i].upper(): i for i in idx}
        if all(a in atoms for a in ("N", "CA", "C", "O")):
            N.append(coords[atoms["N"]])
            CA.append(coords[atoms["CA"]])
            C.append(coords[atoms["C"]])
            O.append(coords[atoms["O"]])
            resids.append(resid)
            chains.append(chain)
            resnames.append(resname)
    if not resids:
        raise ValueError("no residues with a complete N/CA/C/O backbone were found")
    return (
        np.array(N, float), np.array(CA, float), np.array(C, float), np.array(O, float),
        np.array(resids, int), chains, resnames,
    )


def _amide_hydrogens(N, C, O, chains, connected):
    """Place backbone amide H atoms from the previous residue's C=O geometry.

    Returns coordinates with NaN where no hydrogen exists (first residue of a
    chain, or after a chain break) so those residues cannot act as H-bond donors.
    """
    H = np.full_like(N, np.nan)
    co = C - O                                   # reverse of the C=O bond
    norm = np.linalg.norm(co, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        co_unit = co / norm
    for i in range(1, len(N)):
        if connected[i]:
            H[i] = N[i] + co_unit[i - 1]         # 1.0 A along O(i-1)->C(i-1)
    return H


def _hbond_matrix(N, CA, C, O, H):
    """Boolean (R, R) matrix; entry [i, j] true if C=O of i bonds N-H of j."""
    def pdist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    r_on = pdist(O, N)        # acceptor O(i) - donor N(j)
    r_ch = pdist(C, H)
    r_oh = pdist(O, H)
    r_cn = pdist(C, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = _Q1 * _Q2 * _F * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
    ca = pdist(CA, CA)
    bonded = (energy < _HBOND_CUTOFF) & (ca < _CA_CUTOFF)
    bonded &= ~np.isnan(energy)                  # donors without an H are excluded
    np.fill_diagonal(bonded, False)
    return bonded


def assign(molecule: Molecule) -> SecondaryStructure:
    """Assign secondary structure to a protein with a simplified DSSP.

    Returns a :class:`SecondaryStructure` with one code per backbone residue.
    Raises ``ValueError`` if the molecule lacks the metadata or backbone atoms
    needed (e.g. a bare ``.xyz`` file).
    """
    N, CA, C, O, resids, chains, resnames = _backbone_residues(molecule)
    R = len(resids)
    chain_arr = np.array(chains)

    # Chain connectivity: same chain and a real peptide bond to the previous residue.
    connected = np.zeros(R, dtype=bool)
    if R > 1:
        cn = np.linalg.norm(C[:-1] - N[1:], axis=1)
        connected[1:] = (chain_arr[1:] == chain_arr[:-1]) & (cn < _CHAIN_BREAK)

    H = _amide_hydrogens(N, C, O, chains, connected)
    hb = _hbond_matrix(N, CA, C, O, H)

    def same_chain_turn(i, n):
        return i + n < R and chain_arr[i] == chain_arr[i + n]

    # n-turns: an H-bond from residue i to i+n within one chain.
    turn = {n: np.zeros(R, dtype=bool) for n in (3, 4, 5)}
    for n in (3, 4, 5):
        for i in range(R - n):
            if same_chain_turn(i, n) and hb[i, i + n]:
                turn[n][i] = True

    masks = {code: np.zeros(R, dtype=bool) for code in _PRIORITY}

    # Helices: two consecutive n-turns. 4 -> H (alpha), 3 -> G, 5 -> I.
    for n, code, span in ((4, "H", 4), (3, "G", 3), (5, "I", 5)):
        for i in range(1, R):
            if turn[n][i] and turn[n][i - 1]:
                masks[code][i:i + span] = True

    # Bridges -> strands. Needs i-1, i+1, j-1, j+1 in range and |i-j| > 2.
    bridged = np.zeros(R, dtype=bool)
    for i in range(1, R - 1):
        for j in range(i + 3, R - 1):
            parallel = (hb[i - 1, j] and hb[j, i + 1]) or (hb[j - 1, i] and hb[i, j + 1])
            anti = (hb[i, j] and hb[j, i]) or (hb[i - 1, j + 1] and hb[j - 1, i + 1])
            if parallel or anti:
                bridged[i] = bridged[j] = True
    for i in range(R):
        if bridged[i]:
            neighbour = (i > 0 and bridged[i - 1]) or (i < R - 1 and bridged[i + 1])
            masks["E" if neighbour else "B"][i] = True

    # Turns: residues spanned by an n-turn.
    for n in (3, 4, 5):
        for i in np.nonzero(turn[n])[0]:
            masks["T"][i + 1:i + n] = True

    # Bends: sharp kink in the CA trace (virtual angle > 70 degrees).
    if R > 4:
        v1 = CA[2:-2] - CA[:-4]
        v2 = CA[4:] - CA[2:-2]
        cos = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
        )
        masks["S"][2:-2] = np.degrees(np.arccos(np.clip(cos, -1, 1))) > 70.0

    # Resolve by priority (lowest first so higher-priority codes overwrite).
    codes = np.full(R, "-", dtype="<U1")
    for code in reversed(_PRIORITY):
        codes[masks[code]] = code

    return SecondaryStructure(codes, resids, chains, resnames)


def per_atom_ss(molecule: Molecule) -> list:
    """SS code for every atom (its residue's code; ``'-'`` for non-protein atoms)."""
    ss = assign(molecule)
    by_residue = {(c, int(r)): code for c, r, code in zip(ss.chains, ss.resids, ss.codes)}
    chains = molecule.chains or [""] * len(molecule)
    return [by_residue.get((chains[i], int(molecule.resids[i])), "-")
            for i in range(len(molecule))]


@dataclass(frozen=True)
class BackboneTorsions:
    """Per-residue backbone dihedral angles in degrees, in chain/residue order.

    ``phi``/``psi``/``omega`` are ``(R,)`` arrays aligned with ``resids``/``chains``.
    Angles are ``NaN`` where they are undefined: phi at a chain's first residue,
    psi/omega at its last, and across chain breaks.
    """

    resids: np.ndarray
    chains: list
    phi: np.ndarray
    psi: np.ndarray
    omega: np.ndarray

    def __len__(self) -> int:
        return len(self.resids)


def _dihedrals(p0, p1, p2, p3) -> np.ndarray:
    """Vectorised dihedral (degrees) for stacks of four points, each ``(K, 3)``."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=1, keepdims=True) * b1n
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1n, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def backbone_torsions(molecule: Molecule) -> BackboneTorsions:
    """Backbone phi/psi/omega torsions per residue (see :class:`BackboneTorsions`).

    Reuses the same N/CA/C/O backbone extraction as :func:`assign`, so it needs a
    protein with backbone atoms (PDB/mmCIF, not a bare ``.xyz``).
    """
    N, CA, C, O, resids, chains, _ = _backbone_residues(molecule)
    R = len(resids)
    chain_arr = np.array(chains)

    # connected[i]: residue i is peptide-bonded to i-1 (same chain, real C-N bond).
    connected = np.zeros(R, dtype=bool)
    if R > 1:
        cn = np.linalg.norm(C[:-1] - N[1:], axis=1)
        connected[1:] = (chain_arr[1:] == chain_arr[:-1]) & (cn < _CHAIN_BREAK)

    phi = np.full(R, np.nan)
    psi = np.full(R, np.nan)
    omega = np.full(R, np.nan)
    if R > 1:
        link = connected[1:]  # whether residue i is bonded to i-1, for i in 1..R-1
        # phi(i)   = C(i-1)-N(i)-CA(i)-C(i),     defined when i bonded to i-1
        phi[1:][link] = _dihedrals(C[:-1], N[1:], CA[1:], C[1:])[link]
        # omega(i) = CA(i-1)-C(i-1)-N(i)-CA(i),  defined when i bonded to i-1
        omega[1:][link] = _dihedrals(CA[:-1], C[:-1], N[1:], CA[1:])[link]
        # psi(i)   = N(i)-CA(i)-C(i)-N(i+1),     defined when i+1 bonded to i
        psi[:-1][link] = _dihedrals(N[:-1], CA[:-1], C[:-1], N[1:])[link]

    return BackboneTorsions(resids=resids, chains=chains, phi=phi, psi=psi, omega=omega)
