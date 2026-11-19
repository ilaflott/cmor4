# Contributing to CMOR

## Development setup

Install CMOR in editable mode with the development dependencies, and fetch all project table submodules (the tests use them):

```bash
git clone https://github.com/mauzey1/cmor4.git
cd cmor4
pip install -e ".[dev]"
git submodule update --init --recursive
```

## Testing

Run the test suite from the repository root:

```bash
pytest
```

## Code style

Check code style (these checks run in CI):

```bash
ruff check cmor tests
ruff format cmor tests --check
```

Fix lint problems and format the code:

```bash
ruff check cmor tests --fix
ruff format cmor tests
```

## Documentation

See [docs/README.md](./docs/README.md) for how to build and edit the Sphinx documentation.

## Reporting issues

Report bugs and request features via [GitHub Issues](https://github.com/mauzey1/cmor4/issues).
