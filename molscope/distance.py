"""Dense distance and contact-matrix backends.

The public API stays NumPy-first, but dense pairwise operations can be routed
through optional array backends for users who already have Torch or CuPy
installed. Backend arrays are converted back to NumPy by default so existing
MolScope plotting and analysis code keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from .molecule import Molecule

DenseBackend = Literal["auto", "numpy", "torch", "cupy"]


def distance_matrix(
    coords,
    *,
    backend: DenseBackend = "numpy",
    device: str | None = None,
    dtype=None,
    as_numpy: bool = True,
    unit_cell: Optional[Molecule.UnitCell] = None,
):
    """Return the full dense pairwise distance matrix.

    Parameters
    ----------
    coords:
        ``(N, 3)`` coordinates.
    backend:
        ``"numpy"`` for CPU NumPy, ``"torch"`` for PyTorch CPU/GPU,
        ``"cupy"`` for CuPy CUDA arrays, or ``"auto"`` to prefer an available
        GPU backend and otherwise fall back to NumPy.
    device:
        Backend-specific device such as ``"cuda"``, ``"mps"``, or ``"cpu"``.
        Ignored by NumPy and CuPy.
    dtype:
        Backend dtype. Defaults to float64 for NumPy/CuPy and float32 for Torch.
    as_numpy:
        Convert backend arrays back to NumPy. Set ``False`` to keep a Torch or
        CuPy array for advanced GPU workflows.
    unit_cell:
        Optional :class:`molscope.molecule.UnitCell` for periodic boundary
        conditions. Currently only supported by the ``"numpy"`` backend.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    backend = _resolve_backend(backend, device)
    if backend == "numpy":
        return _numpy_distance_matrix(coords, dtype=dtype, unit_cell=unit_cell)
    if backend == "torch":
        if unit_cell is not None:
             raise NotImplementedError("PBC not yet supported for torch backend")
        return _torch_distance_matrix(coords, device=device, dtype=dtype, as_numpy=as_numpy)
    if backend == "cupy":
        if unit_cell is not None:
             raise NotImplementedError("PBC not yet supported for cupy backend")
        return _cupy_distance_matrix(coords, dtype=dtype, as_numpy=as_numpy)
    raise ValueError(f"unsupported dense backend {backend!r}")


def cdist(
    coords1,
    coords2,
    *,
    backend: DenseBackend = "numpy",
    device: str | None = None,
    dtype=None,
    as_numpy: bool = True,
    unit_cell: Optional[Molecule.UnitCell] = None,
):
    """Return the dense ``(N, M)`` pairwise distance matrix between two sets.

    Parameters
    ----------
    coords1:
        ``(N, 3)`` coordinates.
    coords2:
        ``(M, 3)`` coordinates.
    backend, device, dtype, as_numpy, unit_cell:
        See :func:`distance_matrix`.
    """
    coords1 = np.asarray(coords1, dtype=float).reshape(-1, 3)
    coords2 = np.asarray(coords2, dtype=float).reshape(-1, 3)
    backend = _resolve_backend(backend, device)
    if backend == "numpy":
        coords1 = coords1.astype(dtype or float, copy=False)
        coords2 = coords2.astype(dtype or float, copy=False)
        deltas = coords1[:, None, :] - coords2[None, :, :]
        if unit_cell is not None:
            deltas = _apply_mic(deltas, unit_cell)
        return np.sqrt((deltas ** 2).sum(axis=-1))
    if backend == "torch":
        if unit_cell is not None:
             raise NotImplementedError("PBC not yet supported for torch backend")
        torch = _import_torch()
        torch_device = device or _default_torch_device(torch)
        torch_dtype = dtype or torch.float32
        x1 = torch.as_tensor(coords1, dtype=torch_dtype, device=torch_device)
        x2 = torch.as_tensor(coords2, dtype=torch_dtype, device=torch_device)
        mat = torch.cdist(x1, x2)
        return mat.detach().cpu().numpy() if as_numpy else mat
    if backend == "cupy":
        if unit_cell is not None:
             raise NotImplementedError("PBC not yet supported for cupy backend")
        cp = _import_cupy()
        x1 = cp.asarray(coords1, dtype=dtype or cp.float32)
        x2 = cp.asarray(coords2, dtype=dtype or cp.float32)
        deltas = x1[:, None, :] - x2[None, :, :]
        mat = cp.sqrt((deltas ** 2).sum(axis=-1))
        return cp.asnumpy(mat) if as_numpy else mat
    raise ValueError(f"unsupported dense backend {backend!r}")


def contact_matrix(
    coords,
    *,
    cutoff: float,
    backend: DenseBackend = "numpy",
    device: str | None = None,
    dtype=None,
    as_numpy: bool = True,
    unit_cell: Optional[Molecule.UnitCell] = None,
):
    """Return a dense ``(N, N)`` contact matrix from pairwise distances."""
    if cutoff <= 0.0:
        n = len(np.asarray(coords).reshape(-1, 3))
        return np.zeros((n, n), dtype=float)
    mat = distance_matrix(
        coords, backend=backend, device=device, dtype=dtype, as_numpy=as_numpy,
        unit_cell=unit_cell,
    )
    if _is_numpy_array(mat):
        contacts = (mat < cutoff).astype(float)
        np.fill_diagonal(contacts, 0.0)
        return contacts
    if _is_torch_tensor(mat):
        import torch

        contacts = (mat < cutoff).to(dtype=torch.float32)
        contacts.fill_diagonal_(0.0)
        return contacts
    contacts = (mat < cutoff).astype(mat.dtype)
    import cupy as cp

    cp.fill_diagonal(contacts, 0.0)
    return contacts


def contacts_from_matrix(matrix) -> np.ndarray:
    """Return upper-triangle contact pairs from a dense contact matrix."""
    mat = _to_numpy(matrix)
    i, j = np.nonzero(np.triu(mat > 0, k=1))
    return np.stack([i, j], axis=1).astype(int) if len(i) else np.empty((0, 2), dtype=int)


def find_contacts(
    coords, cutoff: float, unit_cell: Optional[Molecule.UnitCell] = None
) -> np.ndarray:
    """Find atom index pairs (i, j) within a cutoff using a pure-NumPy cell list.

    This provides O(n) scaling and is the primary fallback for large structures
    when scipy is absent.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = len(coords)
    if n < 2 or cutoff <= 0.0:
        return np.empty((0, 2), dtype=int)

    # For PBC, we use a simpler box implementation for now if box is non-orthogonal.
    # Orthogonal boxes are easy to bin.
    is_orthogonal = unit_cell is None or (
        unit_cell.alpha == 90.0 and unit_cell.beta == 90.0 and unit_cell.gamma == 90.0
    )

    if not is_orthogonal:
        # Fallback to dense search for non-orthogonal PBC (for now)
        # to avoid complex oblique cell mapping.
        mat = distance_matrix(coords, unit_cell=unit_cell)
        return contacts_from_matrix(mat < cutoff)

    # 1. Setup the grid
    if unit_cell is not None:
        # Align grid with box
        grid_origin = np.zeros(3)
        box_len = np.array([unit_cell.a, unit_cell.b, unit_cell.c])
        grid_dims = np.ceil(box_len / cutoff).astype(int)
        # Wrap coordinates into [0, L]
        wrapped_coords = coords - box_len * np.floor(coords / box_len)
        bins = np.floor(wrapped_coords / cutoff).astype(int)
        # Handle edge case where coord == box_len
        bins = np.clip(bins, 0, grid_dims - 1)
    else:
        grid_origin = coords.min(axis=0)
        bins = np.floor((coords - grid_origin) / cutoff).astype(int)

    # 2. Sort atoms into cells
    indices = np.lexsort(bins.T)
    sorted_bins = bins[indices]
    sorted_coords = coords[indices]  # Use original coords for distance calc

    # Identify unique cells and their boundaries
    cell_change = np.any(sorted_bins[1:] != sorted_bins[:-1], axis=1)
    cell_starts = np.concatenate(([0], np.where(cell_change)[0] + 1))
    cell_ends = np.concatenate((cell_starts[1:], [n]))
    unique_cells = sorted_bins[cell_starts]

    # Hash table (dictionary) for O(1) cell lookup
    cell_map = {tuple(cell): i for i, cell in enumerate(unique_cells)}

    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))

    pairs = []
    cutoff2 = float(cutoff) ** 2

    for i, cell in enumerate(unique_cells):
        start_i, end_i = cell_starts[i], cell_ends[i]
        coords_i = sorted_coords[start_i:end_i]
        orig_idx_i = indices[start_i:end_i]

        # Self-cell contacts (i < j within the same cell)
        if len(coords_i) > 1:
            idx1, idx2 = np.triu_indices(len(coords_i), k=1)
            # Use original coords but apply MIC
            deltas = coords_i[idx1] - coords_i[idx2]
            if unit_cell is not None:
                deltas = _apply_mic(deltas, unit_cell)
            d2 = np.sum(deltas ** 2, axis=1)
            keep = d2 < cutoff2
            if np.any(keep):
                pairs.append(np.stack([orig_idx_i[idx1[keep]], orig_idx_i[idx2[keep]]], axis=1))

        # Neighbor-cell contacts
        for offset in offsets:
            neigh_cell_coord = cell + offset
            
            # PBC wrapping
            if unit_cell is not None:
                wrapped_cell = tuple(np.mod(neigh_cell_coord, grid_dims).astype(int))
                if wrapped_cell in cell_map:
                    j = cell_map[wrapped_cell]
                    # Avoid double counting: only process if j > i
                    # Or if j == i (self image), but we only check offset != 0.
                    # Actually, with PBC, we might have multiple images.
                    # For simplicity, we check all neighbors and unique at the end.
                    start_j, end_j = cell_starts[j], cell_ends[j]
                    coords_j = sorted_coords[start_j:end_j]
                    orig_idx_j = indices[start_j:end_j]

                    deltas = coords_i[:, None, :] - coords_j[None, :, :]
                    deltas = _apply_mic(deltas, unit_cell)
                    d2 = np.sum(deltas ** 2, axis=2)
                    row, col = np.nonzero(d2 < cutoff2)
                    if len(row):
                        pairs.append(np.stack([orig_idx_i[row], orig_idx_j[col]], axis=1))
            else:
                # Non-PBC: half-space check (optimization)
                if offset[0] < 0 or (
                    offset[0] == 0
                    and (offset[1] < 0 or (offset[1] == 0 and offset[2] < 0))
                ):
                    continue
                    
                neigh_cell = tuple(neigh_cell_coord)
                if neigh_cell in cell_map:
                    j = cell_map[neigh_cell]
                    start_j, end_j = cell_starts[j], cell_ends[j]
                    coords_j = sorted_coords[start_j:end_j]
                    orig_idx_j = indices[start_j:end_j]

                    d2 = np.sum((coords_i[:, None, :] - coords_j[None, :, :]) ** 2, axis=2)
                    row, col = np.nonzero(d2 < cutoff2)
                    if len(row):
                        pairs.append(np.stack([orig_idx_i[row], orig_idx_j[col]], axis=1))



    if not pairs:
        return np.empty((0, 2), dtype=int)

    all_pairs = np.concatenate(pairs)
    # Filter out identity pairs (i == j) that might occur from wrapping
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
    # Ensure i < j
    all_pairs.sort(axis=1)
    # Remove duplicates
    all_pairs = np.unique(all_pairs, axis=0)
    # Sort the list of pairs by first then second index
    return all_pairs[np.lexsort(all_pairs.T)]


def find_contact_count(coords, cutoff: float, unit_cell: Optional[Molecule.UnitCell] = None) -> int:
    """Count atom pairs within a cutoff using a pure-NumPy cell list.

    This provides O(n) scaling and is the primary fallback for large structures
    when scipy is absent.
    """
    # Simply use find_contacts and return len for now for simplicity,
    # as PBC makes manual counting tricky with double-counting.
    return len(find_contacts(coords, cutoff, unit_cell=unit_cell))


def backend_name(backend: DenseBackend = "auto", device: str | None = None) -> str:
    """Return the concrete dense backend that would be used."""
    return _resolve_backend(backend, device)


def _numpy_distance_matrix(
    coords: np.ndarray,
    dtype=None,
    unit_cell: Optional[Molecule.UnitCell] = None
) -> np.ndarray:
    coords = coords.astype(dtype or float, copy=False)
    deltas = coords[:, None, :] - coords[None, :, :]
    if unit_cell is not None:
        deltas = _apply_mic(deltas, unit_cell)
    return np.sqrt((deltas ** 2).sum(axis=-1))


def _apply_mic(deltas: np.ndarray, unit_cell: Molecule.UnitCell) -> np.ndarray:
    """Apply the Minimum Image Convention to coordinate deltas."""
    if unit_cell.alpha == 90.0 and unit_cell.beta == 90.0 and unit_cell.gamma == 90.0:
        # Orthogonal box
        box = np.array([unit_cell.a, unit_cell.b, unit_cell.c])
        return deltas - box * np.round(deltas / box)
    
    # Non-orthogonal: convert to fractional, wrap, convert back
    lattice = unit_cell.lattice_matrix()
    inv_lattice = np.linalg.inv(lattice)
    
    # deltas is (..., 3)
    # fractional = deltas @ inv_lattice
    # Cartesian x = fractional f * lattice L (vectors as rows)
    # So f = x @ inv(L)
    fractional = deltas @ inv_lattice
    fractional -= np.round(fractional)
    return fractional @ lattice




def _torch_distance_matrix(coords, *, device, dtype, as_numpy):
    torch = _import_torch()
    torch_device = device or _default_torch_device(torch)
    torch_dtype = dtype or torch.float32
    x = torch.as_tensor(coords, dtype=torch_dtype, device=torch_device)
    mat = torch.cdist(x, x)
    return mat.detach().cpu().numpy() if as_numpy else mat


def _cupy_distance_matrix(coords, *, dtype, as_numpy):
    cp = _import_cupy()
    x = cp.asarray(coords, dtype=dtype or cp.float32)
    deltas = x[:, None, :] - x[None, :, :]
    mat = cp.sqrt((deltas ** 2).sum(axis=-1))
    return cp.asnumpy(mat) if as_numpy else mat


def _resolve_backend(backend: DenseBackend, device: str | None) -> str:
    if backend not in {"auto", "numpy", "torch", "cupy"}:
        raise ValueError(
            "backend must be 'numpy', 'torch', 'cupy' or 'auto', "
            f"got {backend!r}"
        )
    if backend != "auto":
        return backend
    if device:
        return "torch"
    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            return "torch"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "torch"
    except ImportError:
        pass
    try:
        _import_cupy()
        return "cupy"
    except ImportError:
        return "numpy"


def _default_torch_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The 'torch' dense backend requires PyTorch. Install a platform-"
            "appropriate build first, e.g. pip install torch."
        ) from exc
    return torch


def _import_cupy():
    try:
        import cupy
    except ImportError as exc:
        raise ImportError(
            "The 'cupy' dense backend requires CuPy. Install the CUDA-specific "
            "CuPy package that matches your system."
        ) from exc
    return cupy


def _to_numpy(array) -> np.ndarray:
    if _is_numpy_array(array):
        return np.asarray(array)
    if _is_torch_tensor(array):
        return array.detach().cpu().numpy()
    return _import_cupy().asnumpy(array)


def _is_numpy_array(array) -> bool:
    return isinstance(array, np.ndarray)


def _is_torch_tensor(array) -> bool:
    return array.__class__.__module__.split(".", 1)[0] == "torch"
