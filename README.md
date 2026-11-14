# Junction26

## Commands available to CAD agent

| Command | Purpose |
| --- | --- |
| `cad-sim info` | Show problem specification and constraints as JSON |
| `cad-sim status [--ascii] [--json]` | Run FEA; show metrics, an optional stress map, or JSON |
| `cad-sim cut-circle --x X --y Y --r R` | Cut a circular hole |
| `cad-sim cut-rect --x X --y Y --w W --h H [--fillet R]` | Cut a rectangular or rounded pocket |
| `cad-sim cut-slot --x1 X1 --y1 Y1 --x2 X2 --y2 Y2 --r R` | Cut a rounded slot |
| `cad-sim cut-polygon --points "x,y x,y x,y …" [--fillet R]` | Cut a polygonal pocket |
| `cad-sim undo` / `cad-sim reset` | Undo the last cut / restore stock |
| `cad-sim export [--svg PATH] [--dxf PATH]` | Export vector geometry |

All cuts accept `--mirror-x` and `--mirror-y`, save automatically, and run FEA.
Feedback includes volume budget, total and per-load-case compliance, stress limits,
connectivity, wall thickness, feature violations, hotspots, and low-strain zones.


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
