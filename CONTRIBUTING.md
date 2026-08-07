# Contributing

Thanks for your interest in improving this project. This guide covers the
local setup and the standards a change is expected to meet.

## Local setup

```bash
git clone https://github.com/ChetanKumor/AutoML.git
cd AutoML

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
```

Verify the setup:

```bash
pytest                                   # 86 tests, a few seconds
ruff check . && ruff format --check .
```

## Development workflow

1. Branch off `main`.
2. Make the change, with tests covering it.
3. Run the checks below until they pass.
4. Open a pull request describing what changed and why.

## Checks

Every pull request must pass the same checks CI runs:

```bash
ruff format .          # apply formatting
ruff check --fix .     # lint, autofixing what it can
pytest                 # the full suite
```

An end-to-end smoke test is useful before larger changes:

```bash
python train.py --data data/heart-disease.csv --target target
```

## Standards

**Testing.** New behaviour needs a test. Bug fixes need a test that fails
before the fix. Keep tests hermetic: use the `tmp_path` fixture rather than
writing into the repository, and monkeypatch `_base_models` to keep the model
search fast when the search itself is not what you are testing.

**No data leakage.** The preprocessing pipeline must only ever be fitted on
training data. `build_preprocessor` deliberately returns an *unfitted*
pipeline, and `train_models` splits before fitting it. `tests/test_feature_engineer.py`
and `tests/test_model_trainer.py` guard this; do not weaken those tests.

**No fabricated metrics.** Never hard-code, estimate, or copy benchmark numbers
into documentation. Any figure that appears in the README must be reproducible
by running the code.

**Style.** Type-hint public functions, write Google-style docstrings covering
`Args`/`Returns`/`Raises`, and log through `utils.logging_utils.get_logger`
rather than `print`. Modules acquire loggers; only entrypoints call
`configure_logging`.

**Commits.** Use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `build:`, `chore:`, `style:`.
Keep each commit atomic — one logical change, no unrelated edits.

## Reporting bugs

Open an issue using the bug report template. A minimal dataset or code snippet
that reproduces the problem is the single most helpful thing you can include.
