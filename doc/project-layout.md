# Project layout

| Path | Purpose |
| --- | --- |
| `src/junction26/` | Installable Python package |
| `src/junction26/main.py` | Application entry point |
| `src/agents/example.py` | Sample function with a doctest |
| `test/` | Tests (use `unittest`; runnable with pytest) |
| `pyproject.toml` | Project metadata, command, and uv build configuration |
| `uv.lock` | Dependency versions managed by uv |

Run the application with `uv run junction26`. uv installs the package from `src/` automatically.

Place new tests in `test/` using filenames such as `test_example.py`.

Run `uv run pytest` to execute unittest tests in `test/` and doctests in `test/` and `src/`. pytest is installed as a development dependency. `test/test_example.py` demonstrates a unittest test for the sample function.
