import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from river import datasets, metrics


# Constants
# Replace with your classification model ID
MODEL_ID = "574c2ce5-5435-4258-8935-c5853afc7646"
BASE_URL = f"https://api.kappaml.com/v1/models/{MODEL_ID}"
API_KEY = os.getenv("KAPPAML_API_KEY")


# Reuse connections
session = requests.Session()
session.headers.update({"X-API-Key": API_KEY})


def _encode_categorical_feature(
    feature_name: str,
    feature_value: str,
    categorical_encodings: Dict[str, Dict[str, int]],
) -> float:
    if feature_name not in categorical_encodings:
        categorical_encodings[feature_name] = {}
    if feature_value not in categorical_encodings[feature_name]:
        categorical_encodings[feature_name][feature_value] = len(
            categorical_encodings[feature_name]
        )
    return float(categorical_encodings[feature_name][feature_value])


def load_and_prepare_data(
) -> List[Tuple[Dict[str, float], int]]:
    """Load and preprocess a streaming classification dataset (Phishing).

    Returns a list of (features, target) where features are JSON-serializable
    floats and target is 0/1.
    """
    dataset = datasets.Phishing()
    data: List[Tuple[Dict[str, float], int]] = []

    categorical_encodings: Dict[str, Dict[str, int]] = {}

    for features_raw, label_raw in dataset:
        processed: Dict[str, float] = {}
        for feature_name, feature_value in features_raw.items():
            # Normalize by type to JSON serializable floats
            if isinstance(feature_value, (int, float)):
                processed[feature_name] = float(feature_value)
            elif isinstance(feature_value, bool):
                processed[feature_name] = float(int(feature_value))
            elif isinstance(feature_value, datetime):
                epoch = datetime(1970, 1, 1)
                months_since_1970 = (
                    (feature_value.year - epoch.year) * 12
                    + feature_value.month
                    - epoch.month
                )
                processed[f"{feature_name}_months_since_1970"] = float(
                    months_since_1970
                )
                processed["day_of_week"] = float(feature_value.weekday())
                processed["month"] = float(feature_value.month)
                processed["day_of_month"] = float(feature_value.day)
                processed["quarter"] = float(
                    (feature_value.month - 1) // 3 + 1
                )
                processed["is_weekend"] = float(
                    int(feature_value.weekday() >= 5)
                )
            elif isinstance(feature_value, str):
                processed[
                    f"{feature_name}_encoded"
                ] = _encode_categorical_feature(
                    feature_name,
                    feature_value,
                    categorical_encodings,
                )
            else:
                # Fallback to string encoding for unexpected types
                processed[
                    f"{feature_name}_encoded"
                ] = _encode_categorical_feature(
                    feature_name,
                    str(feature_value),
                    categorical_encodings,
                )

        # Phishing labels are booleans; convert to 0/1
        label = int(label_raw)
        data.append((processed, label))

    return data


def process_row(
    features: Dict[str, float],
    true_label: int,
) -> Tuple[Any, Any]:
    """Send predict and learn requests for a single instance."""
    try:
        pred_response = session.post(
            f"{BASE_URL}/predict",
            json={"features": features},
        )
        if pred_response.status_code == 200:
            pred_json = pred_response.json()
            prediction = pred_json.get("prediction")
            # Normalize predictions to 0/1 floats, when possible
            if isinstance(prediction, bool):
                prediction = int(prediction)
            elif isinstance(prediction, (int, float)):
                prediction = int(round(float(prediction)))
            else:
                # Unknown type; skip update
                return None, None
        else:
            return None, None

        learn_response = session.post(
            f"{BASE_URL}/learn",
            json={"features": features, "target": int(true_label)},
        )
        if learn_response.status_code != 200:
            return None, None

        return prediction, true_label
    except requests.exceptions.RequestException:
        return None, None


def get_api_metrics():
    try:
        metrics_response = session.get(f"{BASE_URL}/metrics")
        if metrics_response.status_code == 200:
            return metrics_response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def flatten_metrics(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
    """Flatten nested metrics dict for display."""
    flattened = []
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                try:
                    if isinstance(sub_value, (int, float)):
                        val = float(sub_value)
                    elif isinstance(sub_value, str):
                        try:
                            val = float(sub_value)
                        except (ValueError, TypeError):
                            val = str(sub_value)
                    else:
                        val = str(sub_value)
                except Exception:
                    val = str(sub_value)
                flattened.append({"Metric": f"{key}.{sub_key}", "Value": val})
        else:
            try:
                if isinstance(value, (int, float)):
                    val = float(value)
                elif isinstance(value, str):
                    try:
                        val = float(value)
                    except (ValueError, TypeError):
                        val = str(value)
                else:
                    val = str(value)
            except Exception:
                val = str(value)
            flattened.append({"Metric": key, "Value": val})

    df = pd.DataFrame(flattened)
    df["Value"] = df["Value"].apply(
        lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)
    )
    return df


def main():
    st.title("KappaML AutoML Classification Visualization")
    st.write(
        "This app streams the River Phishing dataset to KappaML and "
        "visualizes online classification metrics."
    )

    if not API_KEY:
        st.error("Please set the KAPPAML_API_KEY environment variable")
        st.stop()

    # Initialize state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.streaming = False
        st.session_state.predictions = []
        st.session_state.true_values = []
        st.session_state.data = load_and_prepare_data()
        st.session_state.api_metrics = None

        # Local, incremental metrics
        st.session_state.metrics = {
            "accuracy": metrics.Accuracy(),
            "precision": metrics.Precision(),
            "recall": metrics.Recall(),
            "f1": metrics.F1(),
            "balanced_accuracy": metrics.BalancedAccuracy(),
        }

        # Confusion counts
        st.session_state.tp = 0
        st.session_state.fp = 0
        st.session_state.tn = 0
        st.session_state.fn = 0

    # Sidebar controls
    with st.sidebar:
        st.subheader("Settings")
        update_delay = st.slider(
            "Update Delay (seconds)",
            min_value=0.001,
            max_value=0.5,
            value=0.01,
            step=0.01,
        )

    # Placeholders
    metrics_plot = st.empty()
    confusion_plot = st.empty()
    accuracy_plot = st.empty()
    metrics_tables = st.empty()
    progress_bar = st.progress(0)

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Streaming") and not st.session_state.streaming:
            st.session_state.streaming = True
    with col2:
        if st.button("Stop Streaming"):
            st.session_state.streaming = False
    with col3:
        if st.button("Reset"):
            st.session_state.predictions = []
            st.session_state.true_values = []
            st.session_state.metrics = {
                "accuracy": metrics.Accuracy(),
                "precision": metrics.Precision(),
                "recall": metrics.Recall(),
                "f1": metrics.F1(),
                "balanced_accuracy": metrics.BalancedAccuracy(),
            }
            st.session_state.tp = 0
            st.session_state.fp = 0
            st.session_state.tn = 0
            st.session_state.fn = 0
            st.session_state.streaming = False
            st.session_state.api_metrics = None

    # Streaming loop
    if st.session_state.streaming:
        total_instances = len(st.session_state.data)
        processed = 0

        for features, true_label in st.session_state.data:
            if not st.session_state.streaming:
                break

            prediction, truth = process_row(features, true_label)
            if prediction is None:
                continue

            pred_int = int(prediction)
            true_int = int(truth)

            st.session_state.predictions.append(pred_int)
            st.session_state.true_values.append(true_int)

            # Update incremental metrics
            for metric in st.session_state.metrics.values():
                metric.update(true_int, pred_int)

            # Update confusion counts
            if true_int == 1 and pred_int == 1:
                st.session_state.tp += 1
            elif true_int == 0 and pred_int == 1:
                st.session_state.fp += 1
            elif true_int == 0 and pred_int == 0:
                st.session_state.tn += 1
            elif true_int == 1 and pred_int == 0:
                st.session_state.fn += 1

            processed += 1
            progress_bar.progress(processed / max(1, total_instances))

            # Refresh API metrics periodically
            if processed % 100 == 0:
                st.session_state.api_metrics = get_api_metrics()

            # Metrics bar chart
            balanced_acc_value = (
                st.session_state.metrics["balanced_accuracy"].get()
            )
            metric_values = {
                "Accuracy": st.session_state.metrics["accuracy"].get(),
                "Precision": st.session_state.metrics["precision"].get(),
                "Recall": st.session_state.metrics["recall"].get(),
                "F1": st.session_state.metrics["f1"].get(),
                "Balanced Acc": balanced_acc_value,
            }
            metrics_fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(metric_values.keys()),
                        y=list(metric_values.values()),
                    )
                ]
            )
            metrics_fig.update_layout(
                title="Current Metrics",
                yaxis_title="Value",
                xaxis_title="Metric",
            )
            metrics_plot.plotly_chart(
                metrics_fig,
                use_container_width=True,
            )

            # Confusion matrix heatmap
            confusion_values = np.array(
                [
                    [st.session_state.tn, st.session_state.fp],
                    [st.session_state.fn, st.session_state.tp],
                ]
            )
            confusion_fig = go.Figure(
                data=go.Heatmap(
                    z=confusion_values,
                    x=["Pred 0", "Pred 1"],
                    y=["True 0", "True 1"],
                    colorscale="Blues",
                    showscale=True,
                )
            )
            confusion_fig.update_layout(title="Confusion Matrix")
            confusion_plot.plotly_chart(
                confusion_fig,
                use_container_width=True,
            )

            # Accuracy over time (last 200)
            acc_over_time = []
            correct_count = 0
            recent_preds = st.session_state.predictions[-200:]
            recent_truth = st.session_state.true_values[-200:]
            for idx, (p, t) in enumerate(
                zip(recent_preds, recent_truth), start=1
            ):
                if p == t:
                    correct_count += 1
                acc_over_time.append(correct_count / idx)

            acc_fig = go.Figure()
            acc_fig.add_trace(
                go.Scatter(
                    y=acc_over_time,
                    name="Accuracy (last 200)",
                )
            )
            acc_fig.update_layout(
                title="Accuracy Over Time",
                xaxis_title="Instance",
                yaxis_title="Accuracy",
            )
            accuracy_plot.plotly_chart(acc_fig, use_container_width=True)

            # Tables
            with metrics_tables.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Local Metrics")
                    accuracy_v = f"{metric_values['Accuracy']:.6f}"
                    precision_v = f"{metric_values['Precision']:.6f}"
                    recall_v = f"{metric_values['Recall']:.6f}"
                    f1_v = f"{metric_values['F1']:.6f}"
                    balacc_v = f"{metric_values['Balanced Acc']:.6f}"

                    local_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Accuracy",
                                "Precision",
                                "Recall",
                                "F1",
                                "Balanced Acc",
                            ],
                            "Value": [
                                accuracy_v,
                                precision_v,
                                recall_v,
                                f1_v,
                                balacc_v,
                            ],
                        }
                    )
                    st.dataframe(local_df, hide_index=True)

                with col2:
                    st.subheader("API Metrics")
                    if st.session_state.api_metrics:
                        api_df = flatten_metrics(st.session_state.api_metrics)
                        st.dataframe(api_df, hide_index=True)
                    else:
                        st.write("No API metrics available yet")

            time.sleep(update_delay)


if __name__ == "__main__":
    main()
