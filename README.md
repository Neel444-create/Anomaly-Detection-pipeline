# Financial Anomaly Detection Pipeline

An end-to-end machine learning project that detects potentially fraudulent credit-card transactions and demonstrates how an ML model can be prepared, served, monitored, and evaluated over time.

## 1. Problem

Credit-card fraud is rare compared with legitimate transactions. Because of this class imbalance, accuracy alone can hide poor fraud detection. This project compares models using precision, recall, F1-score, ROC-AUC, and especially PR-AUC.

## 2. What the pipeline does

1. **Validate and prepare data** — check the dataset schema and values, remove exact duplicates from the processed copy, and create features such as `Amount_Log` and `Time_Of_Day`.
2. **Train and compare models** — train class-weighted Logistic Regression and Random Forest baselines using stratified train/validation/test splits.
3. **Track experiments** — log parameters, metrics, and model artifacts with MLflow.
4. **Serve predictions** — expose the selected model through a FastAPI endpoint.
5. **Monitor data drift** — use Population Stability Index (PSI) to flag changes in numerical feature distributions.
6. **Evaluate retraining candidates** — when drift is detected, compare a candidate model with the current model against configured quality thresholds. Registration is possible; promotion and deployment remain manual.
7. **View results** — use a Streamlit dashboard for summary metrics, drift results, and API-backed predictions.

## 3. Architecture

```text
creditcard.csv
    |
    v
Load + validate + preprocess
    |
    v
Train and evaluate baseline models
    |
    v
Select model using validation PR-AUC
    |
    +----> MLflow experiment tracking
    |
    v
Save selected model locally
    |
    v
FastAPI prediction service ----> Streamlit dashboard
    |
    v
PSI drift monitoring
    |
    v
Retraining decision gate (candidate only; manual promotion)
```

## 4. Dataset

The pipeline expects the ULB/Kaggle credit-card fraud dataset at `data/raw/creditcard.csv`. It contains 284,807 transactions, 31 columns, and a binary `Class` target. The dataset is not included in this repository; obtain it separately and place it at that path. Raw data, processed data, local model files, and MLflow output are ignored by Git.

## 5. Models and evaluation

Two class-weighted baseline models are trained:

- **Logistic Regression** with a `StandardScaler` pipeline.
- **Random Forest** with `class_weight="balanced_subsample"`.

The data is split into stratified 60% training, 20% validation, and 20% test subsets. Validation PR-AUC selects the champion model. The selected model is then refit using training plus validation data and evaluated on the held-out test set.

### Recorded baseline result

In the recorded run, Random Forest was selected by validation PR-AUC.

| Test metric | Result |
| --- | ---: |
| Precision | 0.8974 |
| Recall | 0.7368 |
| F1-score | 0.8092 |
| ROC-AUC | 0.9636 |
| PR-AUC | 0.7961 |

Confusion matrix: `[[56643, 8], [25, 70]]`.

These are results from a previous local run, not a guarantee of results on every environment. They may change with data, configuration, split seed, or dependency versions.

## 6. Tools

- **Python, pandas, NumPy, scikit-learn** — data processing and model training.
- **MLflow** — experiment tracking and local model registry.
- **FastAPI + Pydantic** — prediction API and request validation.
- **Streamlit** — monitoring dashboard.
- **PSI** — feature-distribution drift monitoring.
- **Docker** — container packaging.
- **Pytest + Ruff + GitHub Actions** — automated testing, linting, formatting, and CI checks.

## 7. Project structure

```text
api/app.py                 # FastAPI health and prediction endpoints
dashboard/app.py           # Streamlit dashboard
monitoring/drift.py        # PSI drift report
src/data/
  load.py                  # Read source CSV
  validate.py              # Validate schema, values, and duplicates
  preprocess.py            # Prepare model-ready features
  explore.py               # Data exploration helpers
src/models/
  train.py                 # Train, compare, log, and register model
  evaluate.py              # Evaluation helpers
  retrain.py               # Drift-triggered candidate workflow
tests/                     # Data, model, API, drift, and retraining tests
notebooks/                 # Exploratory analysis notebook
Dockerfile                 # Container definition for inference API
.github/workflows/ci.yml   # CI checks
requirements.txt            # Python dependencies
```

## 8. Setup

Create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the dataset at `data/raw/creditcard.csv`. Do not commit the dataset or generated artifacts.

## 9. Run the pipeline

Run these commands from the project root, in order:

```bash
python src/data/preprocess.py
python src/models/train.py
python monitoring/drift.py
python src/models/retrain.py
```

### Explore MLflow runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open http://127.0.0.1:5000.

### Start the prediction API

```bash
uvicorn api.app:app --reload
```

- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

The `POST /predict` endpoint expects `Time`, `V1`–`V28`, and `Amount`. The API uses the same feature transformation logic as training to keep inference features consistent.

### Start the dashboard

In a separate terminal:

```bash
streamlit run dashboard/app.py
```

Set `API_URL` if the dashboard needs to call an API hosted at a different address.

## 10. Docker

Build and run the API container:

```bash
docker build -t financial-anomaly-api:latest .
docker run --rm -p 8000:8000 --name financial-anomaly-api financial-anomaly-api:latest
```

The image is intended for inference and excludes datasets and development outputs. The model artifact is normally generated locally and is intentionally not committed.

## 11. Tests and code quality

```bash
ruff check .
ruff format --check .
pytest -q
```

GitHub Actions runs automated checks on pushes and pull requests, including tests and a Docker image build.

## 12. Drift monitoring and retraining

The drift report compares numerical feature distributions using PSI. Reference quantile bins are fixed, and the default threshold is 0.20.

The retraining workflow is deliberately conservative:
1. Check for feature drift.
2. Skip candidate training if no feature crosses the threshold.
3. Train a candidate when drift is detected.
4. Compare the candidate and current model on the same test split.
5. Require minimum PR-AUC and recall, plus a configured PR-AUC improvement.
6. Register an accepted candidate separately.

This project does **not** automatically replace the champion model or deploy a candidate. A person must review and approve promotion.

## 13. Limitations

- The anonymized public dataset may not represent current real-world fraud patterns.
- PSI detects changes in feature distributions; it does not by itself detect concept drift or prove that model performance has degraded.
- The default classification threshold is not tuned to a specific business cost.
- Monitoring uses local files rather than a live event stream.
- This is a portfolio project, not a production financial-fraud service.

## 14. Possible next steps

- Select a classification threshold based on the cost of missed fraud and false alarms.
- Add probability calibration and model explanations.
- Log predictions and use delayed labels to measure live model performance.
- Add scheduled monitoring and a documented model approval workflow.
- Separate runtime dependencies from development dependencies to reduce the inference image size.

## Interview summary

“I built an end-to-end credit-card anomaly detection pipeline. I validate and preprocess the data, compare Logistic Regression and Random Forest using metrics appropriate for imbalanced classification, track experiments in MLflow, and serve the selected model with FastAPI. I also added PSI-based drift monitoring and a gated retraining workflow, with manual approval before a candidate can replace the current model. The goal was to demonstrate not just model training, but the main engineering steps needed to operate and monitor an ML model.”
