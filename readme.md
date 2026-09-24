# olist-late-delivery

Late-delivery prediction for a Brazilian e-commerce (Olist) dataset — MLOps
Task 3: turning the Task 2 notebooks into a production inference service.

> Training happens in the notebooks / DVC pipeline stages. The API and the
> `src/inference` package never fit anything — they load the fitted
> transformers and the registered model and predict.

## Status

This is an in-progress build of Task 3. Implemented so far: repository
structure, configuration, notebook-to-module refactor (data prep, labeling,
split, EDA, feature engineering, training), DVC pipeline, MLflow tracking &
registry integration. **Not yet implemented**: Great Expectations suites,
the FastAPI app, Docker/Compose, CI/CD, pytest suite, monitoring endpoints —
tracked as next steps below.

## Repository structure

```
.
├── app/                      # FastAPI service (routes, schemas)
├── config/
│   └── config.yaml           # single source of truth for paths & parameters
├── data/
│   ├── raw/                  # (DVC-tracked) untouched source extracts, if any
│   ├── interim/              # (DVC-tracked) ml_orders_dataset / _labeled
│   └── processed/            # (DVC-tracked) train/val/test + feature tables
├── models/
│   ├── feature_engineering_artifacts/  # (DVC-tracked) fitted imputers/encoders
│   └── model_artifacts/                # (DVC-tracked) local model copy + reports
├── reports/
│   └── eda/                  # (DVC-tracked) EDA plots, summary CSVs, findings
├── notebooks/                # original Task 2 notebooks (kept for reference)
├── src/
│   ├── data/                 # DB access, raw->ML table join, labeling (Notebook 1, 2)
│   ├── features/             # derived features, shipping pressure, fit/transform (Notebook 5)
│   ├── pipeline/             # training-time orchestration: split, EDA, feature eng, train (Notebook 3, 4, 5, 6)
│   ├── inference/            # INFERENCE-ONLY: model loading + predict pipeline
│   └── utils/                # config loader, logging, MLflow helpers
├── tests/                    # FastAPI integration/contract tests
├── requirements/
│   ├── base.txt               # runtime deps (used in the Docker image)
│   ├── pipeline.txt           # + matplotlib/seaborn, for running the DVC training pipeline (incl. EDA)
│   └── dev.txt                # + testing/lint/format/pre-commit (local & CI only)
├── dvc.yaml                  # DVC pipeline: data_prep -> split -> {eda, feature_engineering -> train}
├── pyproject.toml            # ruff/black/mypy/pytest config
└── .env.example              # environment variable template (copy to .env)
```

## Why each tool is here

- **YAML config (`config/config.yaml`)** — every path/parameter used by
  `src/` and `app/` is read from here (or overridden via `CONFIG__*` env
  vars). No hardcoded paths or params anywhere in the code.
- **DVC** — versions the data artifacts (`ml_orders_dataset.parquet`,
  splits, feature tables) and the pipeline that produces them (`dvc.yaml`),
  so any model or metric can be traced back to the exact data that produced
  it.
- **MLflow** — tracks every training run's parameters/metrics/artifacts, and
  registers the selected model with a version and a stage. The inference
  service loads the model from the **registry**, not a local notebook
  folder — see `src/inference/model_loader.py`.
- **EDA as a pipeline stage (`src/pipeline/eda*.py`)** — the same schema,
  missing-value, distribution, and label-relation analysis that justified
  the feature set is regeneratable on demand (`python -m src.pipeline.run_eda`)
  instead of living only inside a notebook, so it stays reproducible and
  diffable as the data changes.
- **Great Expectations** (planned) — validates incoming data before it
  reaches the model (column types, ranges, allowed categories, missing
  rates).
- **FastAPI** (planned) — the prediction service: health, model info,
  single predict, batch predict.
- **pytest** (planned) — unit tests for preprocessing/feature functions,
  data tests (schema/leakage), model tests, API integration tests.
- **Docker / Docker Compose** (planned) — one-command startup of the DB,
  MLflow, the API, and artifact storage.
- **GitHub Actions / CI** (planned) — lint, tests, build & push on every
  push; failing tests stop the pipeline.

## Setup — from zero

### 1. Clone and create a virtual environment

```bash
git clone <repo-url> olist-late-delivery
cd olist-late-delivery
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Runtime only (what the Docker image installs):
pip install -r requirements/base.txt

# To run the training/EDA pipeline (adds matplotlib/seaborn):
pip install -r requirements/pipeline.txt

# Local development (adds tests/lint/format/pre-commit on top of pipeline.txt):
pip install -r requirements/dev.txt
pre-commit install
```

### 3. Configure environment variables

```bash
cp .env.example .env
# edit .env with real DB / MLflow / storage values
```

### 4. Bring up supporting services

Once `docker-compose.yml` is added (see Status above), this will be:

```bash
docker compose up -d db mlflow minio
```

For now, run Postgres and an MLflow tracking server locally, matching the
connection details in `config/config.yaml` / `.env`.

### 5. Run the training pipeline (via DVC)

```bash
dvc repro
```

This runs, in order: `data_prep` → `split` → `{eda, feature_engineering}` →
`train` (see `dvc.yaml`; `eda` is a diagnostic branch off `split` and is not
a dependency of `feature_engineering`/`train`). Each stage is also runnable
individually, e.g.:

```bash
python -m src.pipeline.run_data_prep
python -m src.pipeline.run_split
python -m src.pipeline.run_eda                  # writes reports/eda/*
python -m src.pipeline.run_feature_engineering
python -m src.pipeline.run_train
```

The `train` stage logs every run to MLflow and registers the selected model.
The `eda` stage requires `requirements/pipeline.txt` (matplotlib/seaborn).

### 6. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Run tests

```bash
pytest
```

## Notebooks vs. modules

The original Task 2 notebooks (`notebooks/task2nb1.ipynb` … `task2nb6.ipynb`)
are kept for reference and reproducibility narrative, but no longer contain
the logic used in production — every cell's logic has been moved into
`src/` as functions/modules (see the module docstrings, each of which names
the notebook and cell it refactors).

`notebooks/task2nb4.ipynb` (EDA) is refactored into
`src/pipeline/eda*.py` + `src/pipeline/run_eda.py`. It is diagnostic
output, not a dependency of the model pipeline: `feature_engineering` and
`train` never import from it, and `src/inference` never does either. One
correction was made versus the original notebook's feature-selection table:
`order_status` is documented there as excluded from the final feature set
for label-leakage reasons (see `src/data/labeling.py` and
`src/pipeline/eda_feature_selection.py`), matching what `config.yaml`'s
`features.*` lists actually contain; the two shipping-pressure features
added after the original EDA are also documented there for completeness.

## Next steps

See `config/config.yaml`'s `validation`, `api`, and `mlflow` sections for
the scaffolding already in place for: Great Expectations suites, the
FastAPI app, and monitoring. Docker/Compose, CI/CD, and the pytest suite are
the remaining Definition-of-Done items from Task 3.

## FastAPI Task 3

The service exposes:

- `GET /health` — service/model health.
- `GET /model` and `GET /model/info` — model name, registry stage, source, version, threshold, and feature count.
- `POST /predict` — one validated order.
- `POST /predict/batch` — 1–100 validated orders.
- `GET /docs` — Swagger UI with request/response examples.
- `GET /redoc` — ReDoc documentation.

Run locally with `uvicorn app.main:app --reload --port 8000`, then open `http://127.0.0.1:8000/docs`.

The API rejects unknown fields and invalid values with HTTP 422. Inference never fits transformers or the model; it uses the already-fitted artifacts and the registered MLflow model through `src/inference`.
