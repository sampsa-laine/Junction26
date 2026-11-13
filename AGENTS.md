
## Coding guidelines

Agents should write program code only in the folder called `src/agents` unless the user specifically instructs otherwise.

Put any configuration parameters or sensitive information in the `.env` file. The `.env.example` file should function as a template that is shared through git for filling in the `.env` file. Non-sensitive parameters can be filled in the example file.


## Documentation

Before searching code, check the [table of contents](./doc/table-of-contents.md) for an existing doc on the feature — read it first instead of re-deriving from code.

Agents should always update the documentation on features they implement in the [doc](./doc/) folder. This documentation should be easy to understand for humans but not too long or digressive. Remember to update the [table of contents](./doc/table-of-contents.md).

Agents are free to write any notes, summaries or plans they deem useful for other agents or humans in the [doc/agent-docs](./doc/agent-docs/) folder.

Keep documentation short and simple. Prefer a table or a short list over paragraphs. Avoid duplication


## Code conventions

Prefer better names over comments. Delete a comment that only repeats the code.


## Running Python

The main target is currently called `junction26` and it can be run as follows.

``` shell
uv run junction26
```

New targets can be defined in `pyproject.toml`, under `[project.scripts]`:

``` toml
[project.scripts]
junction26 = "junction26.__init__:main"
newtarget = "junction26.module:targetfunction"
```

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