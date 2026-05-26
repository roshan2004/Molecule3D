"""Optional Gemmi-backed CIF/mmCIF validation helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CifValidationReport:
    """Summary of CIF/mmCIF syntax, atom-site and dictionary validation."""

    path: str
    valid: bool
    syntax_ok: bool
    atom_site_ok: bool
    n_blocks: int = 0
    n_atom_site_rows: int = 0
    dictionary_checked: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        """Raise ``ValueError`` when the report is not valid."""
        if not self.valid:
            raise ValueError("; ".join(self.errors) or "CIF validation failed")


def validate_cif(
    path: str,
    *,
    require_atom_site: bool = True,
    dictionaries=None,
) -> CifValidationReport:
    """Validate CIF/mmCIF syntax and atom-site coordinate columns with Gemmi.

    If ``dictionaries`` is supplied, MolScope shells out to ``gemmi validate
    -d`` for dictionary-aware validation. Dictionary validation requires both
    the Gemmi command-line program and local dictionary files.
    """
    gemmi = _require_gemmi()
    errors, warnings = [], []
    syntax_ok = atom_site_ok = False
    n_blocks = n_atom_site_rows = 0

    try:
        doc = gemmi.cif.read_file(path)
        syntax_ok = True
        n_blocks = len(doc)
    except Exception as exc:
        errors.append(f"syntax parse failed: {exc}")
        return CifValidationReport(
            path=path,
            valid=False,
            syntax_ok=False,
            atom_site_ok=False,
            errors=errors,
        )

    atom_site_errors, n_atom_site_rows = _validate_atom_site(doc)
    if atom_site_errors:
        errors.extend(atom_site_errors)
    atom_site_ok = not atom_site_errors
    if require_atom_site and n_atom_site_rows == 0:
        atom_site_ok = False
        errors.append("no _atom_site coordinate loop found")

    dictionary_checked = False
    if dictionaries is not None:
        dictionary_checked = True
        dict_errors, dict_warnings = _run_gemmi_dictionary_validation(path, dictionaries)
        errors.extend(dict_errors)
        warnings.extend(dict_warnings)

    valid = syntax_ok and atom_site_ok and not errors
    return CifValidationReport(
        path=path,
        valid=valid,
        syntax_ok=syntax_ok,
        atom_site_ok=atom_site_ok,
        n_blocks=n_blocks,
        n_atom_site_rows=n_atom_site_rows,
        dictionary_checked=dictionary_checked,
        errors=errors,
        warnings=warnings,
    )


def _require_gemmi():
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "Gemmi is required for CIF validation; install it with "
            'pip install "molscope[cif]"'
        ) from exc
    return gemmi


def _validate_atom_site(doc) -> tuple[list[str], int]:
    errors: list[str] = []
    total_rows = 0
    required = {
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.type_symbol",
    }
    for block in doc:
        col = block.find_loop("_atom_site.Cartn_x")
        if not col:
            continue
        loop = col.get_loop()
        tags = set(loop.tags)
        missing = sorted(required - tags)
        if missing:
            errors.append(f"block {block.name}: missing atom-site column(s): {', '.join(missing)}")
            continue

        width = loop.width()
        values = list(loop.values)
        tag_index = {tag: i for i, tag in enumerate(loop.tags)}
        rows = loop.length()
        total_rows += rows
        for row_idx in range(rows):
            offset = row_idx * width
            for tag in ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"):
                value = values[offset + tag_index[tag]]
                try:
                    float(value)
                except ValueError:
                    errors.append(
                        f"block {block.name}: row {row_idx + 1} has non-numeric {tag}: {value!r}"
                    )
    return errors, total_rows


def _run_gemmi_dictionary_validation(path: str, dictionaries) -> tuple[list[str], list[str]]:
    gemmi_exe = shutil.which("gemmi")
    if gemmi_exe is None:
        return ["dictionary validation requested but the gemmi command was not found"], []

    dicts = [dictionaries] if isinstance(dictionaries, str) else list(dictionaries)
    cmd = [gemmi_exe, "validate"]
    for dic in dicts:
        cmd.extend(["-d", dic])
    cmd.append(path)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    messages = [line for line in output.splitlines() if line.strip()]
    if result.returncode == 0:
        return [], messages
    return messages or [f"gemmi validate exited with status {result.returncode}"], []
