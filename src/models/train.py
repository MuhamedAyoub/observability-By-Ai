#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for VAE-LSTM Anomaly Detection on CPU/RAM usage data
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

# Import the VAE-LSTM model (assuming it's in vae_lstm_model.py)
from src.models.model import VAE_LSTM, TimeSeriesDataset, train_model, find_threshold, detect_anomalies, plot_results


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load a dataset based on its name
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        DataFrame with loaded data
    """
    # Popular datasets for CPU/RAM usage anomaly detection
    datasets = {
        'nasa': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv',
        'aws': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv',
        'yahoo': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv',
        'art_daily': 'https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/art_daily_jumpsup.csv',
        'art_flatline': 'https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/art_daily_flatmiddle.csv'
    }
    
    if dataset_name in datasets:
        url = datasets[dataset_name]
        print(f"Loading dataset '{dataset_name}' from {url}")
        df = pd.read_csv(url)
        
        # Basic data cleaning and preparation
        
        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Different datasets have different value column names
        value_columns = [col for col in df.columns if col.lower() in ['value', 'cpu', 'usage', 'utilization']]
        if value_columns:
            # If the dataset has multiple value columns, keep all of them
            # Otherwise, rename the single value column to 'value'
            if len(value_columns) == 1:
                df.rename(columns={value_columns[0]: 'value'}, inplace=True)
        else:
            print("Warning: Could not identify value column. Using the first numeric column.")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df.rename(columns={numeric_cols[0]: 'value'}, inplace=True)
            else:
                raise ValueError("No numeric columns found in the dataset.")
        
        return df
    elif dataset_name.endswith('.csv'):
        # Load from local CSV file
        if os.path.exists(dataset_name):
            print(f"Loading dataset from local file: {dataset_name}")
            df = pd.read_csv(dataset_name)
            return df
        else:
            raise FileNotFoundError(f"File {dataset_name} not found.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(datasets.keys())}")


def train_and_evaluate(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[VAE_LSTM, float, Dict]:
    """
    Train and evaluate the VAE-LSTM model on the given dataset
    
    Args:
        df: Input DataFrame with time series data
        args: Command line arguments
    
    Returns:
        Trained model, anomaly threshold, and training history
    """
    # Extract features (use all numeric columns)
    feature_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Using features: {feature_cols}")
    
    # Create dataset
    dataset = TimeSeriesDataset(
        df[feature_cols], 
        seq_length=args.seq_length, 
        train_split=args.train_split
    )
    
    # Get input size from dataset
    input_size = dataset.n_features
    print(f"Input size: {input_size}")
    
    # Create model
    model = VAE_LSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        seq_length=args.seq_length,
        num_layers=args.num_layers
    )
    
    print(f"Model created with parameters:")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Latent dim: {args.latent_dim}")
    print(f"  - Sequence length: {args.seq_length}")
    print(f"  - Number of LSTM layers: {args.num_layers}")
    
    # Get dataloaders
    train_loader = dataset.get_train_dataloader(batch_size=args.batch_size)
    test_loader = dataset.get_test_dataloader(batch_size=args.batch_size)
    
    # Train model
    print("\nStarting training...")
    history = train_model(model, train_loader, test_loader, epochs=args.epochs, learning_rate=args.learning_rate)
    
    # Find threshold for anomaly detection
    print(f"\nFinding anomaly threshold at {args.percentile}th percentile...")
    threshold = find_threshold(model, test_loader, percentile=args.percentile)
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    model.eval()
    test_recons = []
    test_errors = []
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            recon_batch, _, _ = model(data)
            _, recon_error = model.predict(data)
            
            test_recons.append(recon_batch.numpy())
            test_errors.append(recon_error.numpy())
    
    # Concatenate results
    test_recons = np.concatenate(test_recons, axis=0)
    test_errors = np.concatenate(test_errors, axis=0)
    
    # Detect anomalies
    anomalies = test_errors > threshold
    print(f"Detected {np.sum(anomalies)} anomalies in test data.")
    
    # Plot results if requested
    if args.plot:
        print("\nPlotting results...")
        
        # For simplicity, just plot the first feature
        feature_idx = 0
        feature_name = feature_cols[feature_idx]
        
        # Original test data
        original_test = dataset.test_tensor.numpy()
        
        # Reshape for easier plotting
        original_flat = original_test[:, :, feature_idx].flatten()
        recon_flat = test_recons[:, :, feature_idx].flatten()
        
        # Create time index
        time_idx = np.arange(len(original_flat))
        
        # Expand anomalies to match the sequence length
        expanded_anomalies = np.zeros_like(original_flat, dtype=bool)
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                start_idx = i * args.seq_length
                end_idx = start_idx + args.seq_length
                expanded_anomalies[start_idx:end_idx] = True
        
        # Plot
        plt.figure(figsize=(15, 8))
        
        # Original data
        plt.plot(time_idx, original_flat, 'b-', label='Original')
        
        # Reconstructed data
        plt.plot(time_idx, recon_flat, 'g-', label='Reconstructed')
        
        # Highlight anomalies
        anomaly_idx = np.where(expanded_anomalies)[0]
        if len(anomaly_idx) > 0:
            plt.plot(time_idx[anomaly_idx], original_flat[anomaly_idx], 'ro', markersize=4, label='Anomalies')
        
        plt.title(f'VAE-LSTM Anomaly Detection - {feature_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot if output directory is provided
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            plt.savefig(os.path.join(args.output_dir, 'test_results.png'))
        
        plt.show()
    
    return model, threshold, history


def save_outputs(model: VAE_LSTM, threshold: float, history: Dict, args: argparse.Namespace):
    """
    Save model, threshold, and training metrics
    
    Args:
        model: Trained VAE-LSTM model
        threshold: Anomaly detection threshold
        history: Dictionary with training history
        args: Command line arguments
    """
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(args.output_dir, 'vae_lstm_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save threshold
        threshold_path = os.path.join(args.output_dir, 'threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(str(threshold))
        print(f"Threshold saved to {threshold_path}")
        
        # Save model config
        config_path = os.path.join(args.output_dir, 'model_config.txt')
        with open(config_path, 'w') as f:
            f.write(f"input_size: {model.input_size}\n")
            f.write(f"hidden_size: {model.hidden_size}\n")
            f.write(f"latent_dim: {model.latent_dim}\n")
            f.write(f"seq_length: {model.seq_length}\n")
            f.write(f"num_layers: {model.num_layers}\n")
        print(f"Model config saved to {config_path}")
        
        # Save training history
        history_path = os.path.join(args.output_dir, 'training_history.csv')
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['test_loss'], label='Test')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['recon_loss'], label='Reconstruction')
        plt.plot(history['kl_loss'], label='KL Divergence')
        plt.title('Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        history_plot_path = os.path.join(args.output_dir, 'training_history.png')
        plt.savefig(history_plot_path)
        print(f"Training history plot saved to {history_plot_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train VAE-LSTM model for time series anomaly detection')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset to use. Can be one of the predefined datasets (nasa, aws, yahoo, art_daily, art_flatline) or a path to a CSV file')
    parser.add_argument('--seq_length', type=int, default=100,
                      help='Length of input sequences')
    parser.add_argument('--train_split', type=float, default=0.8,
                      help='Ratio of data to use for training (0-1)')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                      help='Size of hidden layer in LSTM')
    parser.add_argument('--latent_dim', type=int, default=16,
                      help='Dimension of latent space')
    parser.add_argument('--num_layers', type=int, default=1,
                      help='Number of LSTM layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate for optimizer')
    
    # Anomaly detection parameters
    parser.add_argument('--percentile', type=float, default=95,
                      help='Percentile to use for anomaly threshold')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--plot', action='store_true',
                      help='Plot results')
    
    args = parser.parse_args()
    
    # Load dataset
    df = load_dataset(args.dataset)
    
    # Train and evaluate
    model, threshold, history = train_and_evaluate(df, args)
    
    # Save outputs
    save_outputs(model, threshold, history, args)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()