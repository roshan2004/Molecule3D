# Contributing

## Development setup

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Documentation

```bash
uv sync --group docs
uv run mkdocs serve
uv run mkdocs build --strict
```

Build a PDF copy of the user guide with Pandoc and a LaTeX engine:

```bash
python scripts/build_user_guide_pdf.py
```

The generated file is written to `docs/_build/molscope-user-guide.pdf`.

## Guidelines

- Keep the core package dependency-light.
- Prefer clear, small APIs over framework-specific abstractions.
- Add tests for new readers, descriptors, graph exporters, or coarse-graining behavior.
- Document limitations when behavior is intentionally simple or prototype-oriented.
