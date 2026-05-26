"""Small reproducible MolScope core benchmarks.

The goal is not to compete with specialist libraries. These timings give users
a practical feel for parser, contact-map, and graph-export costs on the bundled
sample structures.
"""

from __future__ import annotations

import gc
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import molscope as ms
from molscope.distance import backend_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
PROTEIN = DATA / "1fqy.pdb"
ENSEMBLE = DATA / "1aml.pdb"
REPEATS = 7


@dataclass(frozen=True)
class Benchmark:
    structure: str
    operation: str
    callable: Callable[[], object]
    notes: str


def measure(func: Callable[[], object], repeats: int = REPEATS) -> list[float]:
    """Return elapsed milliseconds for repeated calls."""
    times = []
    func()  # warm up import caches and any lazy paths
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return times


def fmt_ms(values: list[float]) -> tuple[str, str]:
    return f"{statistics.median(values):.2f}", f"{min(values):.2f}-{max(values):.2f}"


def main() -> None:
    protein = ms.read(PROTEIN)
    ensemble = ms.read_pdb_models(ENSEMBLE)
    amyloid = ensemble[0]

    benchmarks = [
        Benchmark("1fqy", "parse PDB", lambda: ms.read(PROTEIN), "single 1,661-atom model"),
        Benchmark("1aml", "parse PDB models", lambda: ms.read_pdb_models(ENSEMBLE),
                  "20-model NMR ensemble, 11,960 atoms total"),
        Benchmark("1fqy CA", "distance matrix (NumPy)",
                  lambda: protein.alpha_carbons().distance_matrix(backend="numpy"),
                  "226 x 226 dense CA matrix"),
        Benchmark("1fqy", "residue contact map",
                  lambda: protein.contact_map(cutoff=8.0, level="residue"),
                  "226 x 226 residue map"),
        Benchmark("1aml model 1", "residue contact map",
                  lambda: amyloid.contact_map(cutoff=8.0, level="residue"),
                  "40 x 40 residue map"),
        Benchmark("1fqy", "graph export", lambda: protein.to_graph(),
                  "inferred atom-level bonds"),
        Benchmark("1aml model 1", "graph export", lambda: amyloid.to_graph(),
                  "inferred atom-level bonds"),
    ]
    benchmarks.extend(_optional_backend_benchmarks(protein))

    print("# Benchmark output\n")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Platform: {platform.platform()}")
    print(f"- Machine: {platform.machine() or 'unknown'}")
    print(f"- Repeats: {REPEATS} timed runs after one warm-up\n")
    print("| Structure | Operation | Median (ms) | Range (ms) | Notes |")
    print("| --- | --- | ---: | ---: | --- |")
    for item in benchmarks:
        values = measure(item.callable)
        median, span = fmt_ms(values)
        print(f"| {item.structure} | {item.operation} | {median} | {span} | {item.notes} |")


def _optional_backend_benchmarks(protein) -> list[Benchmark]:
    ca = protein.alpha_carbons()
    rows = []
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows.append(Benchmark(
            "1fqy CA",
            f"distance matrix (Torch {device})",
            lambda: ca.distance_matrix(backend="torch", device=device),
            "optional dense backend",
        ))
    except ImportError:
        pass
    try:
        import cupy  # noqa: F401

        rows.append(Benchmark(
            "1fqy CA",
            "distance matrix (CuPy)",
            lambda: ca.distance_matrix(backend="cupy"),
            "optional CUDA dense backend",
        ))
    except ImportError:
        pass
    auto = backend_name("auto")
    if auto != "numpy":
        rows.append(Benchmark(
            "1fqy CA",
            f"distance matrix (auto -> {auto})",
            lambda: ca.distance_matrix(backend="auto"),
            "auto-selected dense backend",
        ))
    return rows


if __name__ == "__main__":
    main()
