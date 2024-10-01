# main.py (Complete Script)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

from model import LSTMModel

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Receipt Count Prediction')
    parser.add_argument('--data', type=str, default='data_daily.csv', help='Path to the daily data CSV file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    args = parser.parse_args()

    # Data Preparation
    data = pd.read_csv(args.data, parse_dates=['# Date'])
    data['Month'] = data['# Date'].dt.to_period('M')
    monthly_data = data.groupby('Month')['Receipt_Count'].sum().reset_index()
    monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
    monthly_counts = monthly_data['Receipt_Count'].values.astype(float)

    # Normalization
    data_min = monthly_counts.min()
    data_max = monthly_counts.max()
    monthly_counts_norm = (monthly_counts - data_min) / (data_max - data_min)

    # Sequence Creation
    seq_length = 3
    X, y = create_sequences(monthly_counts_norm, seq_length)
    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(y).float().unsqueeze(-1)

    # Model Initialization
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'lstm_model.pth')

    # Prediction
    model.eval()
    future_predictions = []
    input_seq = monthly_counts_norm[-seq_length:].tolist()
    for _ in range(12):
        seq_input = torch.tensor(input_seq[-seq_length:]).float().unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(seq_input)
            future_predictions.append(pred.item())
            input_seq.append(pred.item())

    # Denormalization
    future_predictions_denorm = [(pred * (data_max - data_min)) + data_min for pred in future_predictions]
    last_month = monthly_data['Month'].iloc[-1]
    future_dates = pd.date_range(last_month + pd.DateOffset(months=1), periods=12, freq='M')

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data['Month'], monthly_counts, label='Actual')
    plt.plot(future_dates, future_predictions_denorm, label='Predicted', linestyle='--')
    plt.title('Monthly Receipt Counts and Predictions')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print predicted values
    for date, pred in zip(future_dates, future_predictions_denorm):
        print(f'{date.strftime("%Y-%m")}: {int(pred)}')

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == '__main__':
    main()
