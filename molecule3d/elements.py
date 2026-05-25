"""Per-element reference data: CPK colours and covalent radii.

Values cover the elements common in the sample structures; anything missing
falls back to a neutral default so unknown atoms still render and bond.
"""

# CPK colours (normalised RGB), the convention most molecular viewers use.
CPK_COLORS = {
    "H": (1.00, 1.00, 1.00),
    "C": (0.30, 0.30, 0.30),
    "N": (0.10, 0.10, 0.85),
    "O": (0.85, 0.10, 0.10),
    "S": (0.90, 0.80, 0.20),
    "P": (1.00, 0.50, 0.00),
    "F": (0.30, 0.80, 0.30),
    "CL": (0.20, 0.80, 0.20),
    "BR": (0.60, 0.20, 0.10),
    "I": (0.50, 0.10, 0.60),
    "FE": (0.80, 0.40, 0.10),
    "CA": (0.30, 0.70, 0.70),
    "NA": (0.50, 0.20, 0.80),
    "MG": (0.20, 0.60, 0.20),
    "ZN": (0.50, 0.50, 0.60),
}
DEFAULT_COLOR = (0.50, 0.50, 0.50)

# Covalent radii in angstrom (Cordero et al. 2008, rounded). Used to infer bonds.
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "FE": 1.32, "CA": 1.76, "NA": 1.66, "MG": 1.41, "ZN": 1.22,
}
DEFAULT_RADIUS = 0.75


# Standard atomic weights (g/mol). Unknown atoms fall back to 1.0 so that a
# mass-weighted centre over all-unknown elements reduces to the geometric mean.
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
    "P": 30.974, "F": 18.998, "CL": 35.45, "BR": 79.904, "I": 126.904,
    "FE": 55.845, "CA": 40.078, "NA": 22.990, "MG": 24.305, "ZN": 65.38,
}
DEFAULT_MASS = 1.0


def color(element: str):
    """CPK colour for an element symbol (case-insensitive)."""
    return CPK_COLORS.get(element.upper(), DEFAULT_COLOR)


def covalent_radius(element: str) -> float:
    """Covalent radius in angstrom for an element symbol (case-insensitive)."""
    return COVALENT_RADII.get(element.upper(), DEFAULT_RADIUS)


def mass(element: str) -> float:
    """Atomic weight in g/mol for an element symbol (case-insensitive)."""
    return ATOMIC_MASSES.get(element.upper(), DEFAULT_MASS)
