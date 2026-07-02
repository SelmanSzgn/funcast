# Contributing

## Development setup

```bash
git clone https://github.com/ton-username/funcast.git
cd funcast
pip install -e ".[dev,docs]"
pre-commit install
```

## Running tests

```bash
pytest
```

## Building the documentation locally

```bash
cd docs
make html
```

The documentation is then available at `docs/build/html/index.html`.

## Submitting a pull request

1. Fork the repository
2. Create a feature branch : `git checkout -b feat/my-feature`
3. Commit your changes : `git commit -m "feat: my feature"`
4. Push : `git push origin feat/my-feature`
5. Open a pull request on GitHub
