"""Dense distance and contact-matrix backends.

The public API stays NumPy-first, but dense pairwise operations can be routed
through optional array backends for users who already have Torch or CuPy
installed. Backend arrays are converted back to NumPy by default so existing
MolScope plotting and analysis code keeps working.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

DenseBackend = Literal["auto", "numpy", "torch", "cupy"]


def distance_matrix(
    coords,
    *,
    backend: DenseBackend = "numpy",
    device: str | None = None,
    dtype=None,
    as_numpy: bool = True,
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
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    backend = _resolve_backend(backend, device)
    if backend == "numpy":
        return _numpy_distance_matrix(coords, dtype=dtype)
    if backend == "torch":
        return _torch_distance_matrix(coords, device=device, dtype=dtype, as_numpy=as_numpy)
    if backend == "cupy":
        return _cupy_distance_matrix(coords, dtype=dtype, as_numpy=as_numpy)
    raise ValueError(f"unsupported dense backend {backend!r}")


def contact_matrix(
    coords,
    *,
    cutoff: float,
    backend: DenseBackend = "numpy",
    device: str | None = None,
    dtype=None,
    as_numpy: bool = True,
):
    """Return a dense ``(N, N)`` contact matrix from pairwise distances."""
    if cutoff <= 0.0:
        n = len(np.asarray(coords).reshape(-1, 3))
        return np.zeros((n, n), dtype=float)
    mat = distance_matrix(
        coords, backend=backend, device=device, dtype=dtype, as_numpy=as_numpy
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


def backend_name(backend: DenseBackend = "auto", device: str | None = None) -> str:
    """Return the concrete dense backend that would be used."""
    return _resolve_backend(backend, device)


def _numpy_distance_matrix(coords: np.ndarray, dtype=None) -> np.ndarray:
    coords = coords.astype(dtype or float, copy=False)
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((deltas ** 2).sum(axis=-1))


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
