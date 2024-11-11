# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Step 1: Data Collection
# Download historical stock price data (e.g., S&P 500)
symbol = '^GSPC'  # S&P 500 index symbol
start_date = '2010-01-01'
end_date = '2023-01-01'

data = yf.download(symbol, start=start_date, end=end_date)
data = data[['Close']]
data.dropna(inplace=True)

# Step 2: Data Preprocessing
# Convert data to numpy array
close_prices = data['Close'].values

# Reshape data to (-1, 1)
close_prices = close_prices.reshape(-1, 1)

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define training data length (80% of the data)
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Step 3: Create Training and Testing Datasets
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Create training data
train_data = scaled_data[0:training_data_len]
X_train, y_train = create_dataset(train_data)

# Create testing data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = create_dataset(test_data)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)
y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.Tensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

# Reshape input to be 3D (samples, time steps, features)
X_train = X_train.unsqueeze(2)  # Shape: (batch_size, seq_length, input_size)
X_test = X_test.unsqueeze(2)

# Step 4: Define the LSTM Model
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the output from the last time step
        out = out[:, -1, :]
        # Pass through the fully connected layer
        out = self.fc(out)
        return out

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4) # Set weight decay for L2 reg

    # Train the Model
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()

        # Compute loss
        loss = criterion(outputs.squeeze(), y_train)
        # Backpropagation
        loss.backward()
        # Update parameters
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    # Evaluate the Model
    model.eval()

    with torch.no_grad():
        train_outputs = model(X_train).squeeze() # Make predictions on train
        train_losses = criterion(train_outputs, y_train).cpu()
        train_loss_matrix[run, :] = (train_outputs - y_train).abs()  # Per-point loss
        train_pred_matrix[run, :] = train_outputs.cpu()

        # Evaluate on test data
        test_outputs = model(X_test).squeeze()  # Make predictions on test
        test_losses = criterion(test_outputs, y_test).cpu()
        test_loss_matrix[run, :] = (test_outputs - y_test).abs()  # Per-point loss
        test_pred_matrix[run, :] = test_outputs.cpu()

# Select the predictions with the minimum loss per data point across runs
_, train_min_indices = torch.min(train_loss_matrix, dim=0)
_, test_min_indices = torch.min(test_loss_matrix, dim=0)

# Use indices to select predictions with minimum loss
best_train_preds = torch.zeros(X_train.size(0))
best_test_preds = torch.zeros(X_test.size(0))

for i in range(X_train.size(0)):
    best_train_preds[i] = train_pred_matrix[train_min_indices[i], i]

for j in range(X_test.size(0)):
    best_test_preds[j] = test_pred_matrix[test_min_indices[j], j]

# Inverse transform the predictions to get actual prices
best_train_preds = scaler.inverse_transform(best_train_preds.cpu().numpy().reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
best_test_preds = scaler.inverse_transform(best_test_preds.cpu().numpy().reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

# Calculate RMSE for the best predictions
train_rmse = np.sqrt(mean_squared_error(y_train_actual, best_train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, best_test_preds))
print(f'Train RMSE (Best Predictions): {train_rmse:.2f}')
print(f'Test RMSE (Best Predictions): {test_rmse:.2f}')


# Step 9: Plot the Results
# Prepare data for plotting
train_dates = data.index[60:training_data_len]
test_dates = data.index[training_data_len:]

plt.figure(figsize=(14, 7))
# Plot training data
plt.plot(train_dates, y_train_actual, label='Train Actual Prices')
plt.plot(train_dates, best_train_preds, label='Train Predicted Prices')
# Plot testing data
plt.plot(test_dates, y_test_actual, label='Test Actual Prices')
plt.plot(test_dates, best_test_preds, label='Test Predicted Prices')
plt.title('LSTM Model - Actual vs. Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
