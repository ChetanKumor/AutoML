# Robo Data Scientist — AutoML Platform

[![CI](https://github.com/ChetanKumor/AutoML/actions/workflows/ci.yml/badge.svg)](https://github.com/ChetanKumor/AutoML/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Point it at a tabular dataset and it will profile the target, build a
preprocessing pipeline that matches your column types, train and rank a family
of models, and hand back the winner as a self-contained artifact you can serve
predictions from.

Available as a Streamlit app for exploration and as a CLI for reproducible,
scriptable training.

---

## Contents

- [Why this exists](#why-this-exists)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why this exists

Getting a baseline model out of a new tabular dataset is mostly repetitive:
infer whether the problem is classification or regression, clean the columns,
impute, encode, scale, split, try the usual estimators, tune them a little, and
compare. This project automates that loop while keeping the parts that are easy
to get subtly wrong — most importantly, **fitting every transformer on training
data only** so the reported metrics mean something.

## Quickstart

### Install

```bash
git clone https://github.com/ChetanKumor/AutoML.git
cd AutoML

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.10 or newer is required.

### Train from the command line

```bash
python train.py --data data/heart-disease.csv --target target
```

This writes two timestamped files into `saved_models/`: a `model_*.pkl`
artifact and a `leaderboard_*.csv`.

```
usage: train.py [-h] --data DATA [--target TARGET] [--output-dir OUTPUT_DIR]
                [--task-type {classification,regression}]
```

`--target` is optional; it is inferred from the data when omitted. `--task-type`
overrides the inferred task if the heuristic guesses wrong.

### Run the web app

```bash
streamlit run app.py
```

Upload a CSV or Excel file, confirm the suggested target column, and train.
The app shows the ranked leaderboard, lets you download the winning artifact,
and can score a second uploaded file against any saved model.

### Run in Docker

```bash
docker build -t robo-data-scientist .
docker run --rm -p 8501:8501 -v "$PWD/saved_models:/app/saved_models" robo-data-scientist
```

Then open <http://localhost:8501>. The image runs as a non-root user and
exposes a health check on `/_stcore/health`. Mounting `saved_models` keeps
trained artifacts on the host after the container exits.

### Common tasks

A `Makefile` wraps the usual commands — run `make help` to list them.

```bash
make install-dev    # install runtime + dev dependencies
make check          # lint + tests, exactly what CI runs
make train          # train on the bundled dataset
make run            # start the Streamlit app
make docker-build   # build the image
```

### Predict from Python

```python
import pandas as pd
from utils.predict import make_prediction

df = pd.read_csv("new_records.csv")
predictions = make_prediction(df, "saved_models/model_RandomForest_20250101_120000.pkl")
print(predictions["Predicted_Target"])
```

`make_prediction` takes **raw** input — the artifact carries its own fitted
preprocessor, so you do not reproduce the feature engineering by hand.

## How it works

```
Dataset (CSV / Excel)
        │
        ▼
┌───────────────────────┐
│ Validation            │  extension, size, row/column counts, target
│ utils/validation      │  usability — rejected before any work starts
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Target preparation    │  clean currency/percent strings, label-encode
│ utils/data_utils      │  categorical targets, then infer
│ utils/task_inference  │  classification vs. regression
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ TRAIN / TEST SPLIT    │  ◄── happens BEFORE any transformer is fitted
│ stratified when the   │
│ class balance allows  │
└───────────┬───────────┘
            │
   ┌────────┴────────┐
   ▼                 ▼
train fold        test fold
   │                 │
   │  fit_transform  │  transform only
   ▼                 ▼
┌─────────────────────────────────────────┐
│ Preprocessing pipeline                  │
│ utils/feature_engineer                  │
│                                         │
│  1. FeatureTypeCleaner   "$1,200" → 1200│
│  2. RareCategoryGrouper  rare → "Other" │
│  3. OutlierRemover       cap at median  │
│  4. FeatureSelector      drop low-var,  │
│                          correlated     │
│  5. ColumnTransformer                   │
│       numeric → impute, de-skew, scale  │
│       categorical → impute, one-hot     │
└───────────┬─────────────────────────────┘
            ▼
┌───────────────────────┐
│ Model search          │  GridSearchCV over each candidate,
│ utils/model_trainer   │  cross-validated on the training fold
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Evaluation on the     │  ranked leaderboard
│ held-out test fold    │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ ModelArtifact         │  model + preprocessor + encoder + metadata
│ utils/model_artifact  │  one file, one contract
└───────────────────────┘
```

### Candidate models

**Classification (10)** — Logistic Regression, Decision Tree, Random Forest,
K-Nearest Neighbours, Gaussian Naive Bayes, SVM, Gradient Boosting, XGBoost,
LightGBM, CatBoost. Plus two ensembles built from the models that trained
successfully: a soft-voting ensemble and a stacking ensemble.

**Regression (9)** — Linear Regression, Decision Tree, Random Forest,
K-Nearest Neighbours, SVR, Gradient Boosting, XGBoost, LightGBM, CatBoost.

XGBoost, LightGBM and CatBoost are optional at runtime: if one is not
installed, it is skipped with a warning instead of breaking the run.

### Metrics

| Task | Ranking metric | Also reported |
| --- | --- | --- |
| Classification (balanced) | Accuracy | Balanced accuracy, Precision, Recall, F1 (weighted and macro), confusion matrix |
| Classification (imbalanced) | Balanced accuracy | as above |
| Regression | R² | MSE, RMSE, MAE |

When the least-frequent class falls below half the frequency of the most
common one, ranking switches from accuracy to **balanced accuracy**. Accuracy
rewards guessing the majority class — on a 95/5 split a model that always
answers "no" scores 0.95 while learning nothing, where balanced accuracy scores
it 0.50. `TrainingResult.ranking_metric` records which metric was used.

All figures are computed on the held-out test fold, never on training data.

For every tuned model the leaderboard also carries the cross-validated score
from the hyperparameter search, the selected hyperparameters, and the
**CV−Test gap**. The gap is the cheapest overfitting signal available: a model
that scores well across the training folds but drops on the held-out fold has
a large positive gap, and is worth distrusting even if it ranks highly.

## Project structure

```
AutoML/
├── app.py                        Streamlit UI
├── train.py                      CLI training entrypoint
├── utils/
│   ├── constants.py              Environment-driven configuration
│   ├── logging_utils.py          Centralised logging
│   ├── validation.py             Input checks
│   ├── data_utils.py             Loading, target detection and cleaning
│   ├── task_inference.py         Classification vs. regression inference
│   ├── feature_engineer.py       Custom transformers, pipeline builder
│   ├── model_trainer.py          Model search, evaluation, leaderboard
│   ├── model_artifact.py         Serialization contract
│   └── predict.py                Inference
├── tests/                        131 tests
├── notebooks/                    Exploratory analysis
├── data/                         Sample dataset
└── .github/workflows/ci.yml      Lint, test, smoke, app-boot
```

## Configuration

Every path and tunable is read from the environment, so the same code runs
locally, in CI and in a container.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTOML_MODEL_DIR` | `saved_models` | Where artifacts are written |
| `AUTOML_ENCODER_DIR` | `saved_encoders` | Where encoders are written |
| `AUTOML_LOG_DIR` | `logs` | Log file location |
| `AUTOML_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |
| `AUTOML_LOG_TO_FILE` | `1` | Set to `0` to log to console only |
| `AUTOML_MAX_UPLOAD_MB` | `200` | Upload size ceiling |
| `AUTOML_RANDOM_STATE` | `42` | Global seed |
| `AUTOML_TEST_SIZE` | `0.2` | Held-out fraction |

## Development

```bash
pip install -r requirements-dev.txt

pytest                       # 86 tests
pytest --cov=utils           # with coverage (currently 88%)
ruff check . && ruff format --check .
```

CI runs lint, the test suite on Python 3.10/3.11/3.12, an end-to-end
train-then-predict smoke test, and a headless Streamlit boot check.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Design decisions

**The split happens before the fit.** Fitting imputers, scalers, power
transforms, outlier detectors or feature selectors on the full dataset leaks
information from the evaluation fold and inflates every metric. `build_preprocessor`
therefore returns an *unfitted* pipeline, and `train_models` splits first and
fits only on the training fold. Two tests pin this behaviour.

**One serialization contract.** The model, its preprocessor, the label encoder
and the task type travel together in a single `ModelArtifact`. An earlier
version persisted a bare tuple whose shape the training and inference code each
hard-coded separately; they drifted, and every prediction crashed. Loading a
payload that is not a `ModelArtifact` now fails with an explicit message.

**Column routing is resolved at fit time.** The `ColumnTransformer` selects
columns by dtype rather than from a captured list, because `FeatureSelector`
runs before it and may drop columns.

**Optional heavy dependencies degrade gracefully.** A missing gradient-boosting
backend removes one candidate from the search rather than breaking the import.

## Limitations

Worth knowing before you rely on it:

- **Tabular data only.** No text, image, time-series or multi-label support.
  Time-ordered data will be shuffled by the random split.
- **In-memory.** The dataset must fit in RAM; there is no out-of-core path.
- **Small hyperparameter grids.** `GridSearchCV` sweeps a deliberately narrow
  grid to keep runs quick. Expect a strong baseline, not a tuned champion.
- **Imbalance is detected, not corrected.** Ranking switches to balanced
  accuracy on skewed targets, but no class weighting or resampling is applied,
  so the models themselves are still trained on the skewed distribution.
- **Target inference is heuristic.** It distinguishes discrete codes from
  continuous quantities using cardinality and magnitude, which is a guess.
  Override it with `--target` and `--task-type`.
- **Artifacts are pickles.** Only load artifacts you produced or trust; pickle
  executes code on load.
- **No authentication.** The Streamlit app is unauthenticated — do not expose
  it publicly with sensitive data.
- **No experiment tracking yet.** Runs write a leaderboard CSV; there is no
  MLflow or W&B integration.

## Roadmap

- [ ] Experiment tracking (MLflow) with run comparison
- [ ] Randomised and Bayesian search as alternatives to grid search
- [ ] SHAP-based feature importance in the leaderboard
- [ ] Regression ensembles (voting/stacking, as classification already has)
- [ ] Class weighting and resampling for imbalanced targets (detection and
      imbalance-aware ranking are in place; correction is not)
- [ ] ROC-AUC and PR-AUC in the leaderboard
- [ ] FastAPI inference service and a Dockerfile
- [ ] Time-series aware splitting
- [ ] Data drift detection against the training distribution

## License

[MIT](LICENSE) © Chetan Kumor
