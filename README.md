
# Junction 26 Project

This file currently contains useful information and cheat sheets on how to develop the project and is aimed at humans.
This could perhaps be moved to `CONTRIBUTING.md` once an actual description of the project is written.

Note: All filenames not starting with `~` or `/` are relative to project root `/[...]/junction26/`.


## Installing `uv`

`uv` can be installed via the command below, or using another [listed method](https://docs.astral.sh/uv/getting-started/installation/).

``` shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Any `uv` commands should be run in an (improper) subdirectory of project root.


## Running Python

The main target is currently called `junction26` and it can be run as follows.

``` shell
uv run junction26
```

`uv` managed dependencies are resolved, installed and updated automatically.

New targets can be defined in `pyproject.toml`, under `[project.scripts]`:

``` toml
[project.scripts]
junction26 = "junction26.__init__:main"
newtarget = "junction26.module:targetfunction"
```

Here, `junction26.module` is a module path, which will start with `junction26` given the target is defined in `src/`.
However, any module path is fine, and running `newtarget` is equivalent to running `import sys; from junction26.module import newtarget; sys.exit(newtarget())` in python.


## Testing Python

Tests are written as [`unittest`](https://docs.python.org/3/library/unittest.html) tests under `test/`.
[Doctests](https://docs.python.org/3/library/doctest.html) may be included in docstrings to provide examples, but brevity should be maintained.
Both may be run using `pytest`.

``` shell
uv run pytest
```


## Formatting Python

Run the following command to format all python files in the project.

``` shell
uv run ruff format
```


## Type checking

Use `pyright`.

``` shell
uv run pyright
```


## Pre-commit checklist

- [ ] `ruff`
- [ ] `pyright`
- [ ] `pytest`

