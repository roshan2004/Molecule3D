"""Inter-chain interfaces and ligand-binding-site contacts.

Two related protein-analysis tools, both built on the per-atom chain/residue
metadata and the ``hetero`` (ATOM vs HETATM) flag that ``Molecule`` carries:

* **Interfaces** -- which residues of one chain contact another chain
  (:func:`interface_residues`, :func:`chain_contact_matrix`).
* **Binding sites** -- which protein residues surround a ligand HETATM group
  (:func:`ligands`, :func:`binding_site`).

    mol.interface("A", "B")          # residues across the A/B interface
    mol.binding_site(cutoff=4.5)     # protein residues around the bound ligand
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .molecule import Molecule

# Crystallographic solvent and common monatomic ions skipped when auto-detecting
# ligands (they are HETATM but rarely the "ligand" of interest).
WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "H2O", "SOL", "TIP", "TIP3"})
ION_RESNAMES = frozenset({
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO", "CD",
    "HG", "BR", "IOD", "LI", "RB", "CS", "SR", "BA",
})


@dataclass(frozen=True)
class Residue:
    """A residue identity: chain id, residue number, residue name."""

    chain: str
    resid: int
    resname: str

    def __repr__(self) -> str:
        loc = f"{self.chain}:" if self.chain else ""
        return f"{loc}{self.resname or 'RES'}{self.resid}"


@dataclass(frozen=True)
class LigandResidue:
    """A HETATM group (ligand, cofactor, ion, or solvent) and its atom indices."""

    chain: str
    resid: int
    resname: str
    atom_indices: list[int]

    def __len__(self) -> int:
        return len(self.atom_indices)

    def __repr__(self) -> str:
        loc = f"{self.chain}:" if self.chain else ""
        return f"LigandResidue({loc}{self.resname}{self.resid}, {len(self)} atoms)"


@dataclass
class Interface:
    """Residues and atom contacts across a two-chain interface."""

    chain_a: str
    chain_b: str
    cutoff: float
    residues_a: list[Residue]
    residues_b: list[Residue]
    contacts: list[tuple[int, int]]   # (atom index in chain_a, atom index in chain_b)

    @property
    def n_atom_contacts(self) -> int:
        return len(self.contacts)

    def __repr__(self) -> str:
        return (
            f"Interface({self.chain_a}-{self.chain_b}: "
            f"{len(self.residues_a)}+{len(self.residues_b)} residues, "
            f"{self.n_atom_contacts} atom contacts < {self.cutoff} A)"
        )


@dataclass
class ChainContactMatrix:
    """Symmetric counts of inter-chain atom contacts, labelled by chain id."""

    chains: list[str]
    matrix: np.ndarray                # (C, C) int counts; diagonal is 0

    def count(self, chain_a: str, chain_b: str) -> int:
        """Number of atom contacts between two chains."""
        return int(self.matrix[self.chains.index(chain_a), self.chains.index(chain_b)])


@dataclass
class BindingSite:
    """Protein residues surrounding a ligand, closest first.

    ``residues`` and ``min_distances`` are parallel lists ordered by increasing
    distance to the ligand; ``contacts`` are (protein atom, ligand atom) index
    pairs within ``cutoff``. ``residue_atom_indices`` is aligned with
    ``residues`` and contains all polymer atoms in each binding-site residue.
    """

    ligand: LigandResidue
    cutoff: float
    residues: list[Residue]
    min_distances: list[float]
    contacts: list[tuple[int, int]]
    residue_atom_indices: list[list[int]] = field(default_factory=list)

    @property
    def n_atom_contacts(self) -> int:
        """Number of protein-ligand atom pairs within ``cutoff``."""
        return len(self.contacts)

    @property
    def contact_atom_indices(self) -> list[int]:
        """Protein atoms that make at least one ligand contact."""
        return sorted({int(i) for i, _ in self.contacts})

    @property
    def protein_atom_indices(self) -> list[int]:
        """All polymer atoms in the binding-site residues.

        Sites created by older code may not carry ``residue_atom_indices``; in
        that case this falls back to the protein atoms that directly contact the
        ligand.
        """
        if not self.residue_atom_indices:
            return self.contact_atom_indices
        return sorted({int(i) for atoms in self.residue_atom_indices for i in atoms})

    @property
    def residue_contact_counts(self) -> list[int]:
        """Atom-contact counts aligned with ``residues``."""
        counts = [0] * len(self.residues)
        if not self.residue_atom_indices:
            return counts
        atom_to_residue = {
            int(atom): residue_i
            for residue_i, atoms in enumerate(self.residue_atom_indices)
            for atom in atoms
        }
        for protein_atom, _ in self.contacts:
            residue_i = atom_to_residue.get(int(protein_atom))
            if residue_i is not None:
                counts[residue_i] += 1
        return counts

    def to_records(self) -> list[dict[str, object]]:
        """Return table-friendly per-residue binding-site records."""
        contact_counts = self.residue_contact_counts
        return [
            {
                "chain": residue.chain,
                "resid": residue.resid,
                "resname": residue.resname,
                "min_distance": float(distance),
                "n_atom_contacts": int(contact_counts[i]),
            }
            for i, (residue, distance) in enumerate(zip(self.residues, self.min_distances))
        ]

    def to_molecule(self, molecule: Molecule, include_ligand: bool = False) -> Molecule:
        """Return a subset molecule for the site residues, optionally with ligand atoms."""
        indices = self.protein_atom_indices
        if include_ligand:
            indices = sorted({*indices, *[int(i) for i in self.ligand.atom_indices]})
        if not indices:
            return molecule.take(np.array([], dtype=int))
        return molecule.take(indices)

    def __repr__(self) -> str:
        return (
            f"BindingSite({self.ligand.resname}{self.ligand.resid}: "
            f"{len(self.residues)} residues < {self.cutoff} A)"
        )


# -- internals --------------------------------------------------------------


def _cross_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense ``(len(a), len(b))`` Euclidean distances between two atom sets."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _require_residue_metadata(molecule: Molecule) -> None:
    if not molecule.chains:
        raise ValueError("this analysis needs chain information (read from PDB/mmCIF)")
    if len(molecule.resids) == 0:
        raise ValueError("this analysis needs residue information (read from PDB/mmCIF)")


def _unique_residues(molecule: Molecule, atom_indices) -> list[Residue]:
    """Ordered (by chain, resid) unique residues covering the given atoms."""
    resnames = molecule.resnames or [""] * len(molecule)
    seen: dict[tuple, Residue] = {}
    for i in atom_indices:
        i = int(i)
        key = (molecule.chains[i], int(molecule.resids[i]))
        if key not in seen:
            seen[key] = Residue(molecule.chains[i], int(molecule.resids[i]), resnames[i])
    return [seen[k] for k in sorted(seen)]


def _hetero_groups(molecule: Molecule) -> list[LigandResidue]:
    """Every HETATM residue group, unfiltered."""
    if not molecule.hetero:
        return []
    hetero = molecule.hetero
    groups = []
    for idx, resname, resid, chain in molecule.residue_groups():
        if any(hetero[i] for i in idx):
            groups.append(LigandResidue(chain, resid, resname, list(idx)))
    return groups


# -- interfaces -------------------------------------------------------------


def interface_residues(
    molecule: Molecule, chain_a: str, chain_b: str, cutoff: float = 5.0
) -> Interface:
    """Residues of ``chain_a`` and ``chain_b`` with atoms within ``cutoff`` (A)."""
    _require_residue_metadata(molecule)
    chains = np.asarray(molecule.chains)
    idx_a = np.nonzero(chains == chain_a)[0]
    idx_b = np.nonzero(chains == chain_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(f"chains {chain_a!r} and/or {chain_b!r} not found")

    dist = _cross_distances(molecule.coords[idx_a], molecule.coords[idx_b])
    la, lb = np.nonzero(dist < cutoff)
    contacts = [(int(idx_a[i]), int(idx_b[j])) for i, j in zip(la, lb)]
    return Interface(
        chain_a, chain_b, cutoff,
        residues_a=_unique_residues(molecule, idx_a[np.unique(la)]),
        residues_b=_unique_residues(molecule, idx_b[np.unique(lb)]),
        contacts=contacts,
    )


def chain_contact_matrix(molecule: Molecule, cutoff: float = 5.0) -> ChainContactMatrix:
    """Symmetric matrix of inter-chain atom-contact counts (see :class:`ChainContactMatrix`)."""
    _require_residue_metadata(molecule)
    chain_list = molecule.chain_ids()
    chains = np.asarray(molecule.chains)
    coords = molecule.coords
    n = len(chain_list)
    mat = np.zeros((n, n), dtype=int)
    atom_idx = {c: np.nonzero(chains == c)[0] for c in chain_list}
    for a in range(n):
        for b in range(a + 1, n):
            dist = _cross_distances(coords[atom_idx[chain_list[a]]],
                                    coords[atom_idx[chain_list[b]]])
            mat[a, b] = mat[b, a] = int((dist < cutoff).sum())
    return ChainContactMatrix(list(chain_list), mat)


# -- binding sites ----------------------------------------------------------


def ligands(
    molecule: Molecule, exclude_water: bool = True, exclude_ions: bool = True
) -> list[LigandResidue]:
    """HETATM groups that look like ligands, skipping solvent/ions by default."""
    out = []
    for group in _hetero_groups(molecule):
        name = (group.resname or "").upper()
        if exclude_water and name in WATER_RESNAMES:
            continue
        if exclude_ions and name in ION_RESNAMES:
            continue
        out.append(group)
    return out


def _resolve_ligand(molecule: Molecule, ligand) -> LigandResidue:
    groups = _hetero_groups(molecule)
    if not groups:
        raise ValueError("no HETATM groups found; binding-site analysis needs a ligand")
    if ligand is None:
        candidates = ligands(molecule)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "no non-solvent ligand detected; pass ligand=resname or (chain, resid)"
            )
        names = ", ".join(sorted({g.resname for g in candidates}))
        raise ValueError(
            f"multiple ligands present ({names}); specify ligand=resname or (chain, resid)"
        )
    if isinstance(ligand, LigandResidue):
        return ligand
    if isinstance(ligand, tuple) and len(ligand) == 2:
        chain, resid = ligand
        for g in groups:
            if g.chain == chain and g.resid == int(resid):
                return g
        raise ValueError(f"no HETATM group at chain {chain!r} resid {resid}")
    matches = [g for g in groups if g.resname.upper() == str(ligand).upper()]
    if not matches:
        raise ValueError(f"no HETATM group with resname {ligand!r}")
    if len(matches) > 1:
        locs = ", ".join(f"({g.chain}, {g.resid})" for g in matches)
        raise ValueError(f"resname {ligand!r} matches multiple groups: {locs}; pass (chain, resid)")
    return matches[0]


def binding_site(
    molecule: Molecule, ligand=None, cutoff: float = 4.5
) -> BindingSite:
    """Protein residues within ``cutoff`` (A) of a ligand (see :class:`BindingSite`).

    ``ligand`` selects the HETATM group: a resname (e.g. ``"BEN"``), a
    ``(chain, resid)`` pair, or a :class:`LigandResidue`. When omitted, the
    single non-solvent ligand is used (an error is raised if none or several).
    """
    _require_residue_metadata(molecule)
    target = _resolve_ligand(molecule, ligand)
    lig_idx = np.array(target.atom_indices, dtype=int)

    hetero = (
        np.array(molecule.hetero, dtype=bool)
        if molecule.hetero else np.zeros(len(molecule), bool)
    )
    prot_idx = np.nonzero(~hetero)[0]
    if len(prot_idx) == 0:
        raise ValueError("no polymer (ATOM) atoms to form a binding site")

    dist = _cross_distances(molecule.coords[prot_idx], molecule.coords[lig_idx])
    per_atom_min = dist.min(axis=1)
    close = dist < cutoff
    lp, ll = np.nonzero(close)
    contacts = [(int(prot_idx[i]), int(lig_idx[j])) for i, j in zip(lp, ll)]

    resnames = molecule.resnames or [""] * len(molecule)
    residue_atoms: dict[tuple, list[int]] = {}
    for idx, resname, resid, chain in molecule.residue_groups():
        atoms = [int(i) for i in idx if not hetero[i]]
        if atoms:
            residue_atoms[(chain, int(resid), resname)] = atoms

    site_min: dict[tuple, float] = {}
    site_res: dict[tuple, Residue] = {}
    for local_i in np.unique(lp):
        gi = int(prot_idx[local_i])
        key = (molecule.chains[gi], int(molecule.resids[gi]), resnames[gi])
        site_res[key] = Residue(molecule.chains[gi], int(molecule.resids[gi]), resnames[gi])
        site_min[key] = float(per_atom_min[local_i])
    order = sorted(site_min, key=lambda k: site_min[k])
    return BindingSite(
        ligand=target, cutoff=cutoff,
        residues=[site_res[k] for k in order],
        min_distances=[site_min[k] for k in order],
        contacts=contacts,
        residue_atom_indices=[residue_atoms[k] for k in order],
    )
