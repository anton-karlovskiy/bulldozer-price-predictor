# Bulldozer Price Predictor

A **production** end-to-end machine learning system that predicts the sale price
of bulldozers from their characteristics and past sales.

> *How well can we predict the future sale price of a bulldozer, given its
> characteristics and previous examples of how much similar bulldozers have sold for?*

Because the target is a number, this is a **regression** problem — and because we
predict future sales from past ones, it is also a **time-series / forecasting**
problem. The evaluation metric is **root mean squared log error (RMSLE)**, the
metric used by the [Kaggle Bluebook for Bulldozers competition](https://www.kaggle.com/c/bluebook-for-bulldozers/overview).

## Project structure

```
bulldozer-price-predictor/
├── README.md
├── pyproject.toml
├── data/
│   ├── bluebook-for-bulldozers.zip     # bundled dataset (extracted on first run)
│   └── bluebook-for-bulldozers/        # extracted CSVs (git-ignored, ~500MB)
├── images/                             # figures referenced by the v2 notebook
│   ├── supervised-projects-6-step-ml-framework-tools-highlight.png
│   └── structured-data-custom-bulldozer-advertisement.png
└── notebooks/
    ├── end-to-end-bluebook-bulldozer-price-regression-v2.ipynb     # main (detailed) notebook
    └── end-to-end-bluebook-bulldozer-price-regression-video.ipynb  # condensed walkthrough
```

Two notebook variants are included:

- **`...-v2.ipynb`** — the full, detailed notebook: EDA, feature engineering,
  a Scikit-Learn preprocessing pipeline, model training, hyperparameter tuning,
  evaluation and test-set predictions.
- **`...-video.ipynb`** — a condensed, streamlined walkthrough covering the same
  workflow.

Both notebooks load the data from `../data/bluebook-for-bulldozers/` and extract
it automatically from the bundled zip on first run.

## The data

The dataset comes from the [Kaggle Bluebook for Bulldozers competition](https://www.kaggle.com/c/bluebook-for-bulldozers/data)
and is bundled with this repository as `data/bluebook-for-bulldozers.zip` (~72MB).
It is extracted to `data/bluebook-for-bulldozers/` (~500MB) automatically the first
time you run a notebook, so you don't need to download anything. The extracted
folder is git-ignored to keep the repository light.

There are three core files, split by sale date (a time-series split):

| File | Description |
|------|-------------|
| `Train.csv` | Historical sales up to the end of 2011 (~400,000 examples, 50+ attributes, incl. `SalePrice`). |
| `Valid.csv` | Sales from 1 Jan 2012 – 30 Apr 2012 (~12,000 examples). |
| `Test.csv` | Sales from 1 May 2012 – Nov 2012 (~12,000 examples, `SalePrice` withheld — this is what we predict). |
| `TrainAndValid.csv` | `Train` and `Valid` combined into one file. |
| `ValidSolution.csv` | The `SalePrice` values for `Valid.csv`, keyed by `SalesID`. |

## Approach

The project follows a 6-step machine learning modelling framework:

- **Exploratory data analysis (EDA)** — understanding the dataset and its many
  categorical and numerical attributes.
- **Feature engineering** — extracting date parts (year, month, day, day-of-week,
  day-of-year) from `saledate`, and encoding/imputing missing values.
- **Preprocessing pipeline** — a reproducible Scikit-Learn `ColumnTransformer` +
  `Pipeline` combining ordinal/one-hot encoding and median imputation.
- **Model training** — `RandomForestRegressor` (plus a `HistGradientBoostingRegressor`
  comparison) evaluated with MAE, RMSLE and R².
- **Hyperparameter tuning** — searching for the best random forest configuration.
- **Model persistence** — saving the best model with `joblib` and reloading it.
- **Prediction** — generating sale-price predictions on the held-out test set.

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

1. Install `uv` if you don't have it (see the [uv install guide](https://docs.astral.sh/uv/getting-started/installation/)).
2. Sync the environment — this creates a `.venv` and installs all dependencies
   from `pyproject.toml`:

   ```bash
   uv sync
   ```

3. Open the project in **VS Code** or **Cursor** (both have built-in Jupyter
   notebook support), open one of the notebooks in `notebooks/`, and select the
   `.venv` interpreter as the kernel when prompted:

   - `notebooks/end-to-end-bluebook-bulldozer-price-regression-v2.ipynb` (detailed)
   - `notebooks/end-to-end-bluebook-bulldozer-price-regression-video.ipynb` (condensed walkthrough)

   Then run the cells top to bottom. The dataset extracts itself on first run, and
   `ipykernel` is included so the editor can execute the notebooks directly.

> **Note:** `Train.csv` is large (~400k rows), so training the random forest can
> take a few minutes depending on your machine.
