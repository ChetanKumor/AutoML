# Notebooks

Exploratory analysis kept for reference. These are **not** part of the
application: the platform's behaviour lives in `utils/`, `app.py` and
`train.py`, and is covered by the test suite. Nothing here is imported by the
package or exercised in CI.

| Notebook | What it covers |
| --- | --- |
| `heart_disease_exploration.ipynb` | Manual EDA and modelling on the bundled heart-disease dataset: distributions, correlation structure, and a hand-rolled comparison of several classifiers. Predates the automated pipeline and shows the work the platform now does for you. |

## Running them

```bash
pip install -r ../requirements-dev.txt
pip install jupyter matplotlib seaborn
jupyter notebook
```

Paths inside the notebooks are relative to this directory (for example
`../data/heart-disease.csv`), so start Jupyter from here.

## A caveat on the numbers

Figures in these notebooks come from ad-hoc, exploratory runs and were not
produced by the current pipeline. They are not benchmarks, they are not
comparable with the leaderboard `train.py` writes, and they should not be
quoted as results. For reproducible numbers, run:

```bash
python ../train.py --data ../data/heart-disease.csv --target target
```
