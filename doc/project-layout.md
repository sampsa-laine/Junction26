# Project layout

| Path | Purpose |
| --- | --- |
| `src/agents/` | Core CAD/FEA, SIMP, Oracle, Harbor verifier/CLI, and Streamlit Arena modules |
| `tasks/` | Generated Harbor v1.4 benchmark tasks (synced via `uv run harbor-tasks sync`) |
| `jobs/` | Saved Harbor trial runs and `design.json` outputs |
| `test/` | `unittest` and doctest suite runnable via `uv run pytest` |
| `doc/` | Project documentation |
| `pyproject.toml` | Project metadata, CLI script targets, and `uv` dependencies |

See [arena.md](./arena.md) for the `src/agents/` module breakdown and [harbor.md](./harbor.md) for the benchmark suite.
