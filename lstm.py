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

# Define the LSTM Model at the module level
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define the output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the output from the last time step
        out = out[:, -1, :]
        # Pass through the fully connected layer
        out = self.fc(out)
        return out

# Define the create_dataset function at the module level
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def get_lstm_signal():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Data Collection
    symbol = '^GSPC'  # S&P 500 index symbol
    start_date = '2000-01-01'  # Extended start date for larger dataset
    end_date = '2023-12-31'  # Adjust the end date as needed

    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)

    # Data Preprocessing
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    look_back = 60  # Define the look_back window size

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
        print(f"\nRun {run + 1}/{num_runs} \n")

        # Reinitialize the model for each run
        model = LSTMModel().to(device)

        # Define Loss Function and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

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

    # Inverse transform the predictions to get actual prices
    best_train_preds_np = scaler.inverse_transform(best_train_preds.cpu().numpy().reshape(-1, 1))
    y_train_actual = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
    best_test_preds_np = scaler.inverse_transform(best_test_preds.cpu().numpy().reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

    # Calculate RMSE for training and testing data
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, best_train_preds_np))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, best_test_preds_np))
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')

    # Generate LSTM signals
    # If predicted price > today's price, signal = 1 (buy); else, signal = -1 (sell)
    lstm_signals = np.where(best_test_preds_np > y_test_actual, 1, -1)

    # Align signals with dates
    test_dates = data.index[len(data) - len(y_test_actual):]
    lstm_signal_series = pd.Series(lstm_signals.flatten(), index=test_dates)

    return lstm_signal_series, best_test_preds_np, y_test_actual

if __name__ == "__main__":
    # Call the function to generate signals and get predictions
    lstm_signal_series, best_test_preds_np, y_test_actual = get_lstm_signal()

    # Get dates for plotting
    test_dates = lstm_signal_series.index

    # Plot the Results
    plt.figure(figsize=(14, 7))
    # Plot testing data
    plt.plot(test_dates, y_test_actual, label='Actual Prices')
    plt.plot(test_dates, best_test_preds_np, label='Predicted Prices')
    plt.title('LSTM Model - Actual vs. Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
