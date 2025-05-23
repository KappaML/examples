import streamlit as st
import requests
import time
from river.datasets import Restaurants
from river import metrics
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Constants
MODEL_ID = "90f297a0-c4bc-4060-a080-b299840b3066"  # Replace with your regression model ID
BASE_URL = f"https://api.kappaml.com/v1/models/{MODEL_ID}"
API_KEY = os.getenv("KAPPAML_API_KEY")


# Create a session object to reuse connections
session = requests.Session()
session.headers.update({"X-API-Key": API_KEY})

def load_and_prepare_data():
    """Load and prepare the Restaurants dataset."""
    restaurants = Restaurants()
    
    data = []

    # Create dictionaries to track categorical feature encodings
    categorical_encodings = {}
    
    for x, y in restaurants:
        features_processed = {}
        for feat_name, feat_value in x.items():
            if isinstance(feat_value, (int, float)):
                # Numeric values can be used directly 
                features_processed[feat_name] = feat_value
            elif isinstance(feat_value, datetime):
                # Enhanced datetime features
                epoch = datetime(1970, 1, 1)
                features_processed['months_since_1970'] = (feat_value.year - epoch.year) * 12 + feat_value.month - epoch.month
                features_processed['day_of_week'] = feat_value.weekday()  # 0-6 (Monday-Sunday)
                features_processed['month'] = feat_value.month  # 1-12
                features_processed['day_of_month'] = feat_value.day  # 1-31
                features_processed['quarter'] = (feat_value.month - 1) // 3 + 1  # 1-4
                features_processed['is_weekend'] = int(feat_value.weekday() >= 5)  # 0 or 1
            elif isinstance(feat_value, bool):
                features_processed[feat_name] = int(feat_value)
            elif isinstance(feat_value, str):
                # For string values, use label encoding
                if feat_name not in categorical_encodings:
                    categorical_encodings[feat_name] = {}
                
                if feat_value not in categorical_encodings[feat_name]:
                    categorical_encodings[feat_name][feat_value] = len(categorical_encodings[feat_name])
                
                features_processed[feat_name + '_encoded'] = float(categorical_encodings[feat_name][feat_value])
                
                # Special handling for store_id - extract prefix if it exists
                if feat_name == 'store_id' and '_' in feat_value:
                    prefix = feat_value.split('_')[0]
                    if 'store_prefix' not in categorical_encodings:
                        categorical_encodings['store_prefix'] = {}
                    if prefix not in categorical_encodings['store_prefix']:
                        categorical_encodings['store_prefix'][prefix] = len(categorical_encodings['store_prefix'])
                    features_processed['store_prefix_encoded'] = float(categorical_encodings['store_prefix'][prefix])

        # # Add location-based features if both latitude and longitude exist
        # if 'latitude' in x and 'longitude' in x:
        #     # Distance from city center (using Osaka coordinates as reference)
        #     osaka_lat, osaka_lon = 34.6937, 135.5023
            
        #     def haversine_distance(lat1, lon1, lat2, lon2):
        #         R = 6371  # Earth's radius in kilometers
        #         lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        #         dlat = lat2 - lat1
        #         dlon = lon2 - lon1
        #         a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        #         c = 2 * atan2(sqrt(a), sqrt(1-a))
        #         return R * c
            
        #     features_processed['distance_to_osaka_center'] = haversine_distance(
        #         x['latitude'], x['longitude'], osaka_lat, osaka_lon
        #     )
            
        #     # Add grid-based location features (divide area into 10x10 grid)
        #     features_processed['lat_grid'] = int((x['latitude'] - 34.0) * 10)  # Roughly covering Osaka area
        #     features_processed['lon_grid'] = int((x['longitude'] - 135.0) * 10)
        
        data.append((features_processed, y))
    
    return data

def process_row(features, true_label):
    """Process a single row and return prediction and metrics."""
    try:
        # Make prediction request
        pred_response = session.post(
            f"{BASE_URL}/predict",
            json={"features": features}
        )

        if pred_response.status_code == 200:
            pred_data = pred_response.json()
            prediction = pred_data.get('prediction')
            if prediction is None:
                return None, None
        else:
            return None, None
        
        # Learn from this instance
        learn_response = session.post(
            f"{BASE_URL}/learn",
            json={
                "features": features,
                "target": float(true_label)  # Ensure target is float
            },
        )
        
        if learn_response.status_code != 200:
            return None, None
            
        return prediction, true_label
    
    except requests.exceptions.RequestException:
        return None, None

def get_api_metrics():
    """Fetch metrics from the API."""
    try:
        metrics_response = session.get(f"{BASE_URL}/metrics")
        if metrics_response.status_code == 200:
            return metrics_response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def flatten_metrics(metrics_dict):
    """Flatten nested metrics dictionary into key-value pairs."""
    flattened = []
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                try:
                    if isinstance(sub_value, (int, float)):
                        val = float(sub_value)
                    elif isinstance(sub_value, str):
                        # Try to convert string to float if possible
                        try:
                            val = float(sub_value)
                        except (ValueError, TypeError):
                            val = str(sub_value)
                    else:
                        val = str(sub_value)
                except Exception:
                    val = str(sub_value)
                    
                flattened.append({
                    'Metric': f"{key}.{sub_key}",
                    'Value': val
                })
        else:
            try:
                if isinstance(value, (int, float)):
                    val = float(value)
                elif isinstance(value, str):
                    # Try to convert string to float if possible
                    try:
                        val = float(value)
                    except (ValueError, TypeError):
                        val = str(value)
                else:
                    val = str(value)
            except Exception:
                val = str(value)
                
            flattened.append({
                'Metric': key,
                'Value': val
            })
    
    # Create DataFrame with explicit dtypes
    df = pd.DataFrame(flattened)
    
    # Convert the Value column to string for consistent display
    df['Value'] = df['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
    
    return df

def main():
    st.title("KappaML AutoML Regression Visualization")
    st.write("This app demonstrates real-time predictions using KappaML's API on the Restaurants dataset.")
    
    if not API_KEY:
        st.error("Please set the KAPPAML_API_KEY environment variable")
        st.stop()
    
    # Initialize all session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.streaming = False
        st.session_state.predictions = []
        st.session_state.true_values = []
        st.session_state.metrics = {
            'mae': metrics.MAE(),
            'mape': metrics.MAPE(),
            'rmse': metrics.RMSE(),
            'r2': metrics.R2()
        }
        st.session_state.api_metrics = None
        st.session_state.data = load_and_prepare_data()
    
    # Create control panel
    with st.sidebar:
        st.subheader("Settings")
        update_delay = st.slider(
            "Update Delay (seconds)",
            min_value=0.001,
            max_value=0.5,
            value=0.01,
            step=0.01,
            help="Control how fast the visualization updates. Lower values update faster but may be more CPU intensive."
        )
    
    # Create placeholders for plots and metrics
    prediction_plot = st.empty()
    metrics_plot = st.empty()
    metrics_tables = st.empty()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create columns for start and stop buttons
    col1, col2, col3 = st.columns(3)
    
    # Add start, stop and reset buttons
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
                'mae': metrics.MAE(),
                'mape': metrics.MAPE(),
                'rmse': metrics.RMSE(),
                'r2': metrics.R2()
            }
            st.session_state.streaming = False
            st.session_state.api_metrics = None
    
    # Streaming logic
    if st.session_state.streaming:
        total_instances = len(st.session_state.data)
        processed = 0
        
        for features, true_label in st.session_state.data:
            if not st.session_state.streaming:
                break
                
            prediction, true_value = process_row(features, true_label)
            
            if prediction is not None and true_value is not None:
                st.session_state.predictions.append(float(prediction))  # Ensure prediction is float
                st.session_state.true_values.append(float(true_value))  # Ensure true_value is float
                
                # Update metrics
                for metric in st.session_state.metrics.values():
                    metric.update(true_value, prediction)
                
                # Update progress
                processed += 1
                progress_bar.progress(processed / total_instances)
                
                # Fetch API metrics every 100 samples
                if len(st.session_state.predictions) % 100 == 0:
                    st.session_state.api_metrics = get_api_metrics()
                
                # Create prediction plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.predictions[-100:],
                    name='Predictions',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=st.session_state.true_values[-100:],
                    name='True Values',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title='Last 100 Predictions vs True Values',
                    xaxis_title='Instance',
                    yaxis_title='Value'
                )
                prediction_plot.plotly_chart(fig, use_container_width=True)
                
                # Create metrics plot
                metrics_data = {
                    'MAE': [st.session_state.metrics['mae'].get()],
                    'MAPE': [st.session_state.metrics['mape'].get()],
                    'RMSE': [st.session_state.metrics['rmse'].get()],
                    'R2': [st.session_state.metrics['r2'].get()]
                }
                metrics_fig = go.Figure(data=[
                    go.Bar(name=metric, y=value)
                    for metric, value in metrics_data.items()
                ])
                metrics_fig.update_layout(
                    title='Current Metrics',
                    yaxis_title='Value'
                )
                metrics_plot.plotly_chart(metrics_fig, use_container_width=True)
                
                # Display metrics tables
                with metrics_tables.container():
                    col1, col2 = st.columns(2)
                    
                    # Local metrics table
                    with col1:
                        st.subheader("Local Metrics")
                        local_metrics = pd.DataFrame({
                            'Metric': ['MAE', 'MAPE', 'RMSE', 'R2'],
                            'Value': [
                                f"{st.session_state.metrics['mae'].get():.6f}",
                                f"{st.session_state.metrics['mape'].get():.6f}",
                                f"{st.session_state.metrics['rmse'].get():.6f}",
                                f"{st.session_state.metrics['r2'].get():.6f}"
                            ]
                        })
                        st.dataframe(local_metrics, hide_index=True)
                    
                    # API metrics table
                    with col2:
                        st.subheader("API Metrics")
                        if st.session_state.api_metrics:
                            api_metrics_df = flatten_metrics(st.session_state.api_metrics)
                            st.dataframe(api_metrics_df, hide_index=True)
                        else:
                            st.write("No API metrics available yet")
                
                # Add a small delay to make the visualization smoother
                time.sleep(update_delay)

if __name__ == "__main__":
    main() 