import requests
import time
from river.datasets import Restaurants
from river import metrics
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

MODEL_ID = "90f297a0-c4bc-4060-a080-b299840b3066"
BASE_URL = f"https://api.kappaml.com/v1/models/{MODEL_ID}"
# Get API key from https://app.kappaml.com/api-keys and set it as an environment variable
# export KAPPAML_API_KEY="your_api_key_here"
API_KEY = os.getenv("KAPPAML_API_KEY")

# Create a session object to reuse connections
session = requests.Session()
session.headers.update({"X-API-Key": API_KEY})

def load_and_prepare_data():
    """Load and prepare the Restaurants dataset."""
    # Load dataset
    restaurants = Restaurants()
    
    # Create dictionaries to track categorical feature encodings
    categorical_encodings = {}
    
    # Process data
    data = []
    
    for x, y in restaurants:
        # Process features to make them JSON serializable
        features_processed = {}
        for feat_name, feat_value in x.items():
            if isinstance(feat_value, (int, float)):
                # Numeric values can be used directly
                features_processed[feat_name] = float(feat_value)
            elif isinstance(feat_value, datetime):
                # Convert datetime to months since 1970
                epoch = datetime(1970, 1, 1)
                months_since_1970 = (feat_value.year - epoch.year) * 12 + feat_value.month - epoch.month
                features_processed[feat_name] = float(months_since_1970)
            elif isinstance(feat_value, bool):
                # Convert boolean to int (0 or 1)
                features_processed[feat_name] = float(int(feat_value))
            # elif isinstance(feat_value, str):
            #     # For string values, use a consistent integer encoding
            #     if feat_name not in categorical_encodings:
            #         categorical_encodings[feat_name] = {}
                
            #     if feat_value not in categorical_encodings[feat_name]:
            #         # Assign a new index for this category value
            #         categorical_encodings[feat_name][feat_value] = len(categorical_encodings[feat_name])
                
            #     # Use the assigned index as the numerical value
            #     features_processed[feat_name] = float(categorical_encodings[feat_name][feat_value])
        
        # Send processed features
        data.append((features_processed, y))
    
    return data

def process_row(features, true_label, metrics):
    try:
        # Make prediction request
        pred_response = session.post(
            f"{BASE_URL}/predict",
            json={"features": features}
        )

        if pred_response:
            prediction = pred_response.json()['prediction']
        else:
            prediction = None
            print(f"Error making prediction: {pred_response}")
            return False
        
        # Learn from this instance
        learn_response = session.post(
            f"{BASE_URL}/learn",
            json={
                "features": features,
                "target": true_label
            },
        )
        
        if learn_response.status_code != 200:
            print(f"Error learning from instance: {learn_response}")
            return False
            
        learn_response.raise_for_status()
        
        # Update metrics
        if prediction is not None:
            for metric in metrics:
                metric.update(true_label, prediction)
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return False

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    all_data = load_and_prepare_data()
    
    # Initialize metrics tracking
    total = 0
    local_metrics = [
        # metrics.MAE(),
        # metrics.MSE(),
        # metrics.RMSE(),
        # metrics.R2(),
        # metrics.RMSLE()
        metrics.SMAPE()
    ]
    
    print("Starting to stream data to the API...")
    start_time = time.time()
    
    # Number of worker threads
    max_workers = 5
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all rows to the thread pool
        future_to_row = {executor.submit(process_row, x, y, local_metrics): idx 
                        for idx, (x, y) in enumerate(all_data)}
        
        # Process completed futures
        for future in as_completed(future_to_row):
            if future.result():  # If processing was successful
                total += 1
                
                # Print progress every X instances
                if total % 1000 == 0:
                    print(f"\nProcessed {total} instances")
                    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
                    instances_per_second = total / (time.time() - start_time)
                    print(f"Instances per second: {instances_per_second:.2f}")

                    
                    # API Metrics
                    try:
                        metrics_response = session.get(f"{BASE_URL}/metrics")
                        if metrics_response.status_code == 200:
                            api_metrics = metrics_response.json()
                            print("API model metrics:")
                            print(api_metrics)
                            print("\nLocal model metrics:")
                            for metric in local_metrics:
                                print(f"{metric.__class__.__name__}: {metric.get():.4f}")
                    except Exception as e:
                        print(f"Error fetching metrics: {e}")
    
    # Print final results
    if total > 0:
        print(f"\nFinal Results:")
        print(f"Total instances processed: {total}")
        print("\nLocal model metrics:")
        for metric in local_metrics:
            print(f"{metric.__class__.__name__}: {metric.get():.4f}")
        
        # API Metrics
        try:
            metrics_response = session.get(f"{BASE_URL}/metrics")
            if metrics_response.status_code == 200:
                api_metrics = metrics_response.json()
                print("\nAPI model metrics:")
                print(api_metrics)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching final metrics: {e}")
        
        # Time
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")

    # Close the session when done
    session.close()

if __name__ == "__main__":
    main() 