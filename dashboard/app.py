"""Simple Streamlit dashboard for local fraud-model monitoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import streamlit as st

from monitoring.drift import DriftConfig, detect_drift, drift_report, has_drift
from src.data.preprocess import DEFAULT_PROCESSED_DATA_PATH
from src.data.validate import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load processed records used for local monitoring summaries."""
    return pd.read_csv(DEFAULT_PROCESSED_DATA_PATH)


@st.cache_resource
def load_model():
    """Load the local serving artifact once per dashboard process."""
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    """Load local model metadata, returning a safe empty object when absent."""
    return (
        json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if METADATA_PATH.is_file()
        else {}
    )


def post_prediction(payload: dict[str, float]) -> dict:
    """Call the FastAPI serving endpoint with a validated raw transaction."""
    request = Request(
        f"{API_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    """Render a compact dashboard for local model monitoring."""
    st.set_page_config(page_title="Financial Anomaly Monitoring", layout="wide")
    st.title("Financial Anomaly Monitoring")
    st.caption("Local monitoring view — model promotion remains a manual decision.")

    if not DEFAULT_PROCESSED_DATA_PATH.is_file() or not MODEL_PATH.is_file():
        st.error("Run preprocessing and training before opening the dashboard.")
        return

    data = load_data()
    metadata = load_metadata()
    model = load_model()
    feature_names = metadata.get(
        "feature_names", [column for column in data.columns if column != "Class"]
    )
    predictions = model.predict(data[feature_names])
    anomalies = int(predictions.sum())
    anomaly_percentage = (anomalies / len(data) * 100) if len(data) else 0.0
    model_version = metadata.get("mlflow", {}).get("registered_model_version", "local")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Model version", f"v{model_version}")
    metric_columns[1].metric("Total transactions", f"{len(data):,}")
    metric_columns[2].metric("Anomalies", f"{anomalies:,}")
    metric_columns[3].metric("Anomaly percentage", f"{anomaly_percentage:.2f}%")

    st.subheader("Current model metrics")
    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        metric_table = {
            key.replace("_", " ").upper(): value
            for key, value in test_metrics.items()
            if key != "confusion_matrix"
        }
        st.dataframe(pd.DataFrame([metric_table]), use_container_width=True)
    else:
        st.info("No local evaluation metadata found.")

    st.subheader("Feature drift")
    split_index = int(len(data) * 0.8)
    drift_results = detect_drift(
        data.iloc[:split_index], data.iloc[split_index:], config=DriftConfig()
    )
    report = drift_report(drift_results)
    st.dataframe(
        report.style.format({"Drift Score": "{:.4f}"}),
        use_container_width=True,
        height=360,
    )
    if has_drift(drift_results):
        st.warning("Drift detected — candidate retraining can be reviewed.")
    else:
        st.success("No monitored feature crossed the configured PSI threshold.")

    st.subheader("Transaction prediction")
    st.caption(f"Sends the transaction to `{API_URL}/predict`.")
    with st.form("prediction_form"):
        columns = st.columns(3)
        payload: dict[str, float] = {}
        for index, feature in enumerate(FEATURE_COLUMNS):
            with columns[index % 3]:
                minimum = 0.0 if feature in {"Time", "Amount"} else None
                payload[feature] = st.number_input(
                    feature, value=0.0, min_value=minimum, format="%.6f"
                )
        submitted = st.form_submit_button("Score transaction")
    if submitted:
        try:
            result = post_prediction(payload)
            st.success(
                f"Prediction: {result['prediction']} | Anomaly score: {result['anomaly_score']:.6f}"
            )
        except (URLError, TimeoutError, ValueError) as error:
            st.error(f"Prediction API is unavailable: {error}")


if __name__ == "__main__":
    main()
