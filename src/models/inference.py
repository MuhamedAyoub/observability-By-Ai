#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for fetching Prometheus metrics and performing anomaly detection using the VAE-LSTM model
"""

import argparse
import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

# Import the VAE-LSTM model (assuming it's in vae_lstm_model.py)
from src.models.model import VAE_LSTM, PrometheusDataProcessor, detect_anomalies, plot_results


def query_prometheus(query: str, start_time: str, end_time: str, step: str, prometheus_url: str) -> pd.DataFrame:
    """
    Query Prometheus for time series data

    Args:
        query: PromQL query
        start_time: Start time in RFC3339 format or relative (e.g., '1h')
        end_time: End time in RFC3339 format or relative (e.g., 'now')
        step: Query step (e.g., '15s', '1m')
        prometheus_url: URL of Prometheus API

    Returns:
        DataFrame with query results
    """
    # Process relative time specifications
    now = datetime.now()

    if start_time.endswith('h'):
        hours = int(start_time[:-1])
        start_time = (now - timedelta(hours=hours)).isoformat() + 'Z'
    elif start_time.endswith('m'):
        minutes = int(start_time[:-1])
        start_time = (now - timedelta(minutes=minutes)).isoformat() + 'Z'
    elif start_time.endswith('d'):
        days = int(start_time[:-1])
        start_time = (now - timedelta(days=days)).isoformat() + 'Z'

    if end_time == 'now':
        end_time = now.isoformat() + 'Z'

    # Construct the API URL
    url = f"{prometheus_url}/api/v1/query_range"

    # Set up the parameters for the request
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': step
    }

    print(f"Querying Prometheus: {query}")
    print(f"Time range: {start_time} to {end_time} with step {step}")

    # Make the request
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
    except requests.exceptions.RequestException as e:
        print(f"Error querying Prometheus: {e}")
        return pd.DataFrame()

    # Parse the response
    data = response.json()

    # Check if there's an error in the response
    if data['status'] != 'success':
        print(f"Error in Prometheus response: {data.get('error', 'Unknown error')}")
        return pd.DataFrame()

    # Check if there are results
    if not data['data']['result']:
        print("No data returned from Prometheus query")
        return pd.DataFrame()

    # Parse results into a DataFrame
    results = []

    for series in data['data']['result']:
        metric = series['metric']
        metric_name = metric.get('__name__', 'unknown')

        # Add other labels
        labels = {k: v for k, v in metric.items() if k != '__name__'}

        # Extract values and timestamps
        for point in series['values']:
            timestamp, value = point
            result = {
                'timestamp': datetime.fromtimestamp(timestamp),
                'value': float(value),
                'metric': metric_name
            }
            # Add labels
            result.update(labels)
            results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    return df


def process_prometheus_data(dfs: List[pd.DataFrame], feature_columns: List[str]) -> pd.DataFrame:
    """
    Process and combine Prometheus metrics for anomaly detection

    Args:
        dfs: List of DataFrames with Prometheus metrics
        feature_columns: List of columns to keep for features

    Returns:
        Processed DataFrame ready for anomaly detection
    """
    # If there are multiple DataFrames, we need to merge them
    if len(dfs) > 1:
        # Resample all dataframes to a common time frequency
        resampled_dfs = []
        for df in dfs:
            # Resample to 1-minute intervals (adjust as needed)
            df_resampled = df.resample('1min').mean()
            resampled_dfs.append(df_resampled)

        # Merge on index (timestamp)
        merged_df = pd.concat(resampled_dfs, axis=1)

        # Forward fill missing values
        merged_df.fillna(method='ffill', inplace=True)

        # Backward fill any remaining missing values at the beginning
        merged_df.fillna(method='bfill', inplace=True)

        # If there are still missing values, fill with zeros
        merged_df.fillna(0, inplace=True)

        return merged_df
    else:
        # Just return the single DataFrame
        return dfs[0]


def load_model(model_path: str, config_path: str) -> VAE_LSTM:
    """
    Load a trained VAE-LSTM model

    Args:
        model_path: Path to model weights file
        config_path: Path to model config file

    Returns:
        Loaded model
    """
    # Load model config
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            config[key] = int(value)

    # Create model
    model = VAE_LSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        latent_dim=config['latent_dim'],
        seq_length=config['seq_length'],
        num_layers=config['num_layers']
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Set to evaluation mode
    model.eval()

    return model


def detect_prometheus_anomalies(model: VAE_LSTM, df: pd.DataFrame, feature_columns: List[str],
                                threshold: float) -> pd.DataFrame:
    """
    Detect anomalies in Prometheus data

    Args:
        model: Trained VAE-LSTM model
        df: DataFrame with Prometheus metrics
        feature_columns: List of feature columns to use
        threshold: Anomaly threshold

    Returns:
        DataFrame with anomaly flags
    """
    # Create data processor
    processor = PrometheusDataProcessor(seq_length=model.seq_length)

    # Prepare data for inference
    tensor_data, meta = processor.prepare_data(df, feature_columns)

    # Run model inference
    with torch.no_grad():
        recon_batch, _, _ = model(tensor_data)
        _, recon_error = model.predict(tensor_data)

    # Convert to numpy
    recon_error_np = recon_error.numpy()

    # Detect anomalies
    anomalies = recon_error_np > threshold

    # Convert tensor to numpy
    recon_np = recon_batch.numpy()

    # Reshape for inverse transformation
    batch_size, seq_len, n_features = recon_np.shape
    recon_flat = recon_np.reshape(-1, n_features)

    # Inverse transform
    recon_orig = processor.inverse_transform(recon_flat, meta['scaler'])

    # Reshape back
    recon_orig = recon_orig.reshape(batch_size, seq_len, n_features)

    # Map back to original DataFrame
    # For simplicity, we'll take the middle point of each sequence
    middle_idx = model.seq_length // 2

    # Calculate the indices in the original DataFrame
    start_idx = middle_idx
    end_idx = len(df) - middle_idx
    result_indices = range(start_idx, end_idx)

    # Create result DataFrame
    result_data = {
        'timestamp': df.index[result_indices],
    }

    # Add original features and reconstructions
    for i, col in enumerate(feature_columns):
        result_data[col] = df[col].values[result_indices]
        result_data[f'recon_{col}'] = recon_orig[:, middle_idx, i]

    # Add anomaly flag and reconstruction error
    result_data['anomaly'] = anomalies
    result_data['recon_error'] = recon_error_np

    # Create DataFrame
    result_df = pd.DataFrame(result_data)
    result_df.set_index('timestamp', inplace=True)

    return result_df


def plot_anomalies(result_df: pd.DataFrame, feature_columns: List[str], output_dir: str = None):
    """
    Plot original vs reconstructed time series with anomalies highlighted

    Args:
        result_df: DataFrame with original, reconstructed data and anomaly flags
        feature_columns: List of feature columns to plot
        output_dir: Directory to save plots
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Plot each feature
    for feature in feature_columns:
        plt.figure(figsize=(15, 8))

        # Plot original data
        plt.plot(result_df.index, result_df[feature], 'b-', label='Original')

        # Plot reconstructed data
        plt.plot(result_df.index, result_df[f'recon_{feature}'], 'g-', label='Reconstructed')

        # Highlight anomalies
        anomaly_indices = result_df[result_df['anomaly']].index
        if len(anomaly_indices) > 0:
            plt.plot(anomaly_indices, result_df.loc[anomaly_indices, feature], 'ro', markersize=8, label='Anomalies')

        plt.title(f'Anomaly Detection for {feature}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save if output directory is specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'anomaly_{feature}.png'))

        plt.show()

    # Plot reconstruction error
    plt.figure(figsize=(15, 8))
    plt.plot(result_df.index, result_df['recon_error'], 'b-', label='Reconstruction Error')
    plt.axhline(y=np.mean(result_df['recon_error']) + 2 * np.std(result_df['recon_error']),
                color='r', linestyle='--', label='Threshold (Mean + 2*Std)')
    plt.title('Reconstruction Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save if output directory is specified
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'reconstruction_error.png'))

    plt.show()


def save_results(result_df: pd.DataFrame, output_dir: str):
    """
    Save anomaly detection results

    Args:
        result_df: DataFrame with anomaly detection results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full results
    result_df.to_csv(os.path.join(output_dir, 'anomaly_detection_results.csv'))

    # Save anomalies only
    anomalies_df = result_df[result_df['anomaly']]
    anomalies_df.to_csv(os.path.join(output_dir, 'anomalies.csv'))

    # Save summary
    total_points = len(result_df)
    anomaly_points = len(anomalies_df)
    anomaly_percentage = (anomaly_points / total_points) * 100 if total_points > 0 else 0

    summary = {
        'total_points': total_points,
        'anomaly_points': anomaly_points,
        'anomaly_percentage': anomaly_percentage,
        'start_time': str(result_df.index.min()),
        'end_time': str(result_df.index.max())
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Results saved to {output_dir}")
    print(f"Total data points: {total_points}")
    print(f"Detected anomalies: {anomaly_points} ({anomaly_percentage:.2f}%)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Detect anomalies in Prometheus metrics using VAE-LSTM model')

    # Prometheus parameters
    parser.add_argument('--prometheus_url', type=str, default='http://localhost:9090',
                        help='URL of Prometheus API')
    parser.add_argument('--query', type=str, action='append', required=True,
                        help='PromQL query (can specify multiple times for multiple metrics)')
    parser.add_argument('--start_time', type=str, default='1d',
                        help='Start time (RFC3339 or relative, e.g., "1d" for 1 day ago)')
    parser.add_argument('--end_time', type=str, default='now',
                        help='End time (RFC3339 or "now")')
    parser.add_argument('--step', type=str, default='1m',
                        help='Query step (e.g., "15s", "1m")')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--threshold_path', type=str, required=True,
                        help='Path to anomaly threshold file')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='prometheus_anomalies',
                        help='Directory to save outputs')
    parser.add_argument('--plot', action='store_true',
                        help='Plot results')

    args = parser.parse_args()

    # Query Prometheus for metrics
    print("Querying Prometheus for metrics...")
    dfs = []

    for query in args.query:
        df = query_prometheus(
            query=query,
            start_time=args.start_time,
            end_time=args.end_time,
            step=args.step,
            prometheus_url=args.prometheus_url
        )

        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No data retrieved from Prometheus. Exiting.")
        return

    # Process Prometheus data
    print("Processing Prometheus data...")
    feature_columns = [col for col in dfs[0].columns if col not in ['metric']]
    processed_df = process_prometheus_data(dfs, feature_columns)

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.config_path)

    # Load threshold
    with open(args.threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    print(f"Anomaly threshold: {threshold}")

    # Detect anomalies
    print("Detecting anomalies...")
    result_df = detect_prometheus_anomalies(model, processed_df, feature_columns, threshold)

    # Plot results if requested
    if args.plot:
        print("Plotting results...")
        plot_anomalies(result_df, feature_columns, args.output_dir)

    # Save results
    print("Saving results...")
    save_results(result_df, args.output_dir)

    print("Anomaly detection complete!")


if __name__ == "__main__":
    main()