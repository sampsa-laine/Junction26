# Junction26

## Command cheatsheet

| Command | Purpose |
| --- | --- |
| `uv sync` | Install project and development dependencies |
| `uv run junction26` | Run the application |
| `uv run pytest` | Run unittest tests and doctests |
| `uv run pytest -v` | Run tests with individual test names |

See [project layout and testing](doc/project-layout.md) for details.

## Adding `uv` targets

Note `junction26 = "junction26.main:main"` in `pyproject.toml`.
This is just `[targetName] = "[module]:[function]`.
Then this can be run as `uv run [targetName]`.
Note that this is equivalent to running `import sys; from [module] import [function]; sys.exit([function]())` in python.
