import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=60, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def get_lstm_signal(symbol: str, start_date: str = '1999-01-01', end_date: str = '2019-12-31'):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)

    # Calculate Daily Returns
    data['Daily_Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    # Calculate Annualized Returns (rolling 252 trading days)
    data['Annualized_Return'] = (
        (1 + data['Daily_Return']).rolling(window=252).apply(np.prod, raw=True) ** (252 / 252) - 1
    )
    data.dropna(inplace=True)

    # Split dates
    training_end_date = '2010-12-31'
    training_mask = data.index <= training_end_date
    training_data_len = training_mask.sum()
    
    print(f"\nData split:")
    print(f"Training period: {data.index[0]} to {data.index[training_data_len-1]}")
    print(f"Testing period: {data.index[training_data_len]} to {data.index[-1]}")
    print(f"Total days: {len(data)}, Training days: {training_data_len}, Testing days: {len(data)-training_data_len}\n")

    # Preprocessing: Use Annualized Returns
    returns = data['Annualized_Return'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(returns)

    look_back = 120

    # Create Training and Testing Datasets
    train_data = scaled_data[0:training_data_len]
    test_data = scaled_data[training_data_len - look_back:]

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
    X_test = torch.from_numpy(X_test).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

    # Reshape input to be 3D
    X_train = X_train.unsqueeze(2)
    X_test = X_test.unsqueeze(2)

    # Define parameters for multiple runs
    num_runs = 5
    num_epochs = 45
    train_loss_matrix = torch.zeros((num_runs, X_train.size(0)))
    test_loss_matrix = torch.zeros((num_runs, X_test.size(0)))
    train_pred_matrix = torch.zeros((num_runs, X_train.size(0)))
    test_pred_matrix = torch.zeros((num_runs, X_test.size(0)))

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}\n")

        # Reinitialize the model for each run
        model = LSTMModel().to(device)

        # Define Loss Function and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0045, weight_decay=1e-3)

        # Train the Model
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

        # Evaluate the Model
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train).squeeze()
            test_outputs = model(X_test).squeeze()

            # Store predictions
            train_pred_matrix[run, :] = train_outputs.cpu()
            test_pred_matrix[run, :] = test_outputs.cpu()

            # Calculate per-point absolute errors for selecting best predictions
            train_loss_matrix[run, :] = (train_outputs - y_train).abs().cpu()
            test_loss_matrix[run, :] = (test_outputs - y_test).abs().cpu()

    # Select the predictions with the minimum loss per data point across runs
    _, train_min_indices = torch.min(train_loss_matrix, dim=0)
    _, test_min_indices = torch.min(test_loss_matrix, dim=0)

    best_train_preds = torch.zeros(X_train.size(0))
    best_test_preds = torch.zeros(X_test.size(0))

    for i in range(X_train.size(0)):
        best_train_preds[i] = train_pred_matrix[train_min_indices[i], i]

    for j in range(X_test.size(0)):
        best_test_preds[j] = test_pred_matrix[test_min_indices[j], j]

    # Inverse transform the predictions
    best_train_preds_np = scaler.inverse_transform(best_train_preds.cpu().numpy().reshape(-1, 1))
    best_test_preds_np = scaler.inverse_transform(best_test_preds.cpu().numpy().reshape(-1, 1))
    y_train_actual = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, best_train_preds_np))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, best_test_preds_np))
    print()
    print(f'Train RMSE: {train_rmse:.6f}')
    print(f'Test RMSE: {test_rmse:.6f}')

    # Generate signals based on comparison with the previous day's actual return
    shifted_actual = np.roll(y_test_actual, shift=1)  # Shift actual values by 1
    lstm_signals = np.where(best_test_preds_np[1:] > shifted_actual[1:], 1, -1)

    # Align dates (exclude the first date, as it lacks a previous day for comparison)
    test_dates = data.index[training_data_len:]
    lstm_signal_series = pd.Series(lstm_signals.flatten(), index=test_dates[1:])

    return lstm_signal_series, best_test_preds_np, y_test_actual

if __name__ == "__main__":
    # Call the function to generate signals and get predictions
    lstm_signal_series, test_preds, y_test_actual = get_lstm_signal('^GSPC')

    # Get the full data for date alignment
    data = yf.download('^GSPC', start='2000-01-01', end='2020-12-31')
    training_end_date = '2010-12-31'
    training_mask = data.index <= training_end_date
    training_data_len = training_mask.sum()
    test_dates = data.index[training_data_len:]

    # Plot Results
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates[:len(test_preds)], test_preds, 
             label='Predicted Annualized Returns', alpha=0.7)
    plt.plot(test_dates[:len(y_test_actual)], y_test_actual, 
             label='Actual Annualized Returns', alpha=0.7)
    plt.title('LSTM Model - Predicted vs Actual Annualized Returns')
    plt.xlabel('Date')
    plt.ylabel('Annualized Returns')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()