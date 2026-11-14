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

## Using Modal to offload tasks

Set up your modal API key, for example with `uv run modal setup`.
Add `--env modal` to the `[uv run] harbor run ...` command,
among potential other flags coming after `harbor`.

For testing `modal`, `uv run modal run src/junction26/modaltest.py` should work:
```
$ uv run modal run src/junction26/modaltest.py 
✓ Initialized. View run at https://modal.com/apps/joelkronqvist/main/ap-sD8v22gNIoDH9zftLrS4xD
✓ Created objects.
├── ✓ Uploaded /home/school/Dev/Junction26/src/junction26/modaltest.py
└── ✓ Created Function square.
This code is running on a remote worker!
the square is 1764
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/joelkronqvist/main/ap-sD8v22gNIoDH9zftLrS4xD
```
