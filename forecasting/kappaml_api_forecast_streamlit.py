import streamlit as st
import requests
import time
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from river import datasets

# Constants
MODEL_ID = "adea6c81-0bfd-4bd2-bff3-a837f9cb12df"  # Replace with your forecast model ID
BASE_URL = f"https://api.kappaml.com/v1/models/{MODEL_ID}"
API_KEY = os.getenv("KAPPAML_API_KEY")

# Create a session object to reuse connections
session = requests.Session()
session.headers.update({"X-API-Key": API_KEY})

def load_bike_data():
    """Load and prepare the Bikes dataset from River."""
    dataset = datasets.Bikes()
    data = []
    for x, y in dataset:
        data.append({
            'moment': x['moment'],
            'station': x['station'],
            'temperature': x['temperature'],
            'humidity': x['humidity'],
            'wind': x['wind'],
            'clouds': x['clouds'],
            'pressure': x['pressure'],
            'description': x['description'],
            'bikes': y
        })
    return pd.DataFrame(data)

def train_model(features_list):
    """Train the model with historical data."""
    try:
        # Get the latest data point for training
        latest_features = features_list[-1]
        
        # Prepare the training request
        training_request = {
            "features": {
                "temperature": latest_features["temperature"],
                "humidity": latest_features["humidity"],
                "wind": latest_features["wind"],
                "clouds": latest_features["clouds"],
                "pressure": latest_features["pressure"]
            },
            "target": latest_features["value"]
        }
        
        # Print training request/response in a collapsible section
        with st.expander("Debug: Training API Call", expanded=False):
            st.write("Request:")
            with st.container():
                st.json(training_request)
            
            response = session.post(
                f"{BASE_URL}/learn",
                json=training_request
            )
            
            st.write("Response:")
            with st.container():
                if response.status_code == 200:
                    st.json(response.json())
                    return response.json()
                else:
                    st.error(f"Error training model: {response.text}")
                    st.json({"error": response.text})
                    return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred during training: {e}")
        return None

def make_forecast_request(features_dict, horizon=24):
    """Make a forecast request to the KappaML API."""
    try:
        # Generate a list of feature dictionaries for each time step in the horizon
        features = []
        for _ in range(horizon):
            features.append({
                "humidity": features_dict["humidity"],
                "temperature": features_dict["temperature"],
                "wind": features_dict["wind"],
                "clouds": features_dict["clouds"],
                "pressure": features_dict["pressure"]
            })
        
        # Prepare the forecast request
        forecast_request = {
            "features": features,
            "horizon": horizon
        }
        
        # Print forecast request/response in a collapsible section
        with st.expander("Debug: Forecast API Call", expanded=False):
            st.write("Request:")
            with st.container():
                st.json(forecast_request)
            
            response = session.post(
                f"{BASE_URL}/forecast",
                json=forecast_request
            )
            
            st.write("Response:")
            with st.container():
                if response.status_code == 200:
                    st.json(response.json())
                    return response.json()
                else:
                    st.error(f"Error making forecast: {response.text}")
                    st.json({"error": response.text})
                    return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred: {e}")
        return None

def main():
    st.title("KappaML AutoML Bike Sharing Forecasting")
    st.write("This app demonstrates real-time forecasting of bike availability using KappaML's API.")
    
    if not API_KEY:
        st.error("Please set the KAPPAML_API_KEY environment variable")
        st.stop()
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        with st.spinner('Loading bike-sharing dataset...'):
            df = load_bike_data()
            st.session_state.initialized = True
            st.session_state.df = df
            st.session_state.stations = df['station'].unique()
            st.session_state.current_idx = 0
            st.session_state.forecasts = []
            st.session_state.forecast_dates = []
    
    # Create placeholders for plots
    forecast_plot = st.empty()
    metrics_container = st.empty()
    
    # Add controls
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_station = st.selectbox(
            "Select Station",
            options=st.session_state.stations,
            index=0
        )
    with col2:
        horizon = st.slider("Forecast Horizon (hours)", 1, 48, 24)
    with col3:
        update_interval = st.slider("Update Interval (seconds)", 1, 10, 5)
    
    # Add a start button
    if st.button("Start Forecasting"):
        try:
            df = st.session_state.df
            station_data = df[df['station'] == selected_station].copy()
            station_data = station_data.sort_values('moment')
            
            while True:
                if st.session_state.current_idx >= len(station_data):
                    st.warning("Reached end of dataset. Restarting...")
                    st.session_state.current_idx = 0
                
                # Get current data window - use last 24 hours of data
                window_size = 24  # hours
                start_idx = max(0, st.session_state.current_idx - window_size + 1)
                current_data = station_data.iloc[start_idx:st.session_state.current_idx + 1]
                
                # Get the latest data point
                latest_row = current_data.iloc[-1]
                
                # Prepare features for training
                training_features = []
                for _, row in current_data.iterrows():
                    feature = {
                        "timestamp": row['moment'].isoformat(),
                        "value": float(row['bikes']),
                        "temperature": float(row['temperature']),
                        "humidity": float(row['humidity']),
                        "wind": float(row['wind']),
                        "clouds": float(row['clouds']),
                        "pressure": float(row['pressure'])
                    }
                    training_features.append(feature)
                
                # Train the model first
                with st.spinner('Training model...'):
                    train_response = train_model(training_features)
                    if not train_response:
                        st.error("Training failed. Skipping this iteration.")
                        st.session_state.current_idx += 1
                        continue
                
                # Prepare features for forecasting
                forecast_features = {
                    "timestamp": latest_row['moment'].isoformat(),
                    "value": float(latest_row['bikes']),
                    "temperature": float(latest_row['temperature']),
                    "humidity": float(latest_row['humidity']),
                    "wind": float(latest_row['wind']),
                    "clouds": float(latest_row['clouds']),
                    "pressure": float(latest_row['pressure'])
                }
                
                # Make forecast request
                forecast_response = make_forecast_request(forecast_features, horizon)
                
                if forecast_response:
                    # Extract forecast values and dates
                    forecast_values = forecast_response.get('forecast', [])
                    forecast_dates = [
                        latest_row['moment'] + timedelta(hours=i+1)
                        for i in range(len(forecast_values))
                    ]
                    
                    # Create forecast plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=current_data['moment'],
                        y=current_data['bikes'],
                        name='Historical Data',
                        line=dict(color='blue')
                    ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add confidence intervals if available
                    if 'lower_bound' in forecast_response and 'upper_bound' in forecast_response:
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_response['lower_bound'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(255,0,0,0.1)',
                            name='Lower Bound'
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_response['upper_bound'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(255,0,0,0.1)',
                            name='Upper Bound'
                        ))
                    
                    # Add weather information
                    weather_info = f"""
                    Temperature: {latest_row['temperature']:.1f}°C
                    Humidity: {latest_row['humidity']}%
                    Wind: {latest_row['wind']} m/s
                    Clouds: {latest_row['clouds']}%
                    Pressure: {latest_row['pressure']} hPa
                    Weather: {latest_row['description']}
                    """
                    
                    fig.update_layout(
                        title=f'Bike Availability Forecast - Station: {selected_station}',
                        xaxis_title='Time',
                        yaxis_title='Available Bikes',
                        showlegend=True,
                        template='plotly_dark',  # Use dark theme
                        annotations=[
                            dict(
                                x=0.02,
                                y=0.98,
                                xref='paper',
                                yref='paper',
                                text=weather_info,
                                showarrow=False,
                                align='left',
                                bgcolor='rgba(0,0,0,0.5)'
                            )
                        ]
                    )
                    
                    forecast_plot.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast metrics if available
                    if 'metrics' in forecast_response:
                        with metrics_container:
                            st.subheader("Forecast Metrics")
                            metrics_df = pd.DataFrame(forecast_response['metrics'])
                            st.dataframe(metrics_df)
                
                # Move to next data point
                st.session_state.current_idx += 1
                
                # Add a small delay
                time.sleep(update_interval)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main() 