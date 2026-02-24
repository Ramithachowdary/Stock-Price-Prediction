# Stock-Price-Prediction using RNN


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Stock price prediction is a challenging time series forecasting problem. This project builds a Recurrent Neural Network (RNN) model using PyTorch to predict stock closing prices based on historical data. The model learns sequential patterns from past prices to forecast future values.
### Dataset 
1.trainset.csv — Historical stock data used for training <br>
2.testset.csv — Stock data used for evaluating the model <br>
3.Feature used: Close (Closing Price)<br>
4.Data is normalized using MinMaxScaler fitted only on training data to avoid data leakage<br>
5.Sequences of length 60 are created as input windows to predict the next price<br>

## Design Steps

### Step 1: Data Loading and Preprocessing
The training and test datasets are loaded from CSV files and closing prices are extracted and normalized using MinMaxScaler fitted only on training data. Sliding window sequences of length 60 are created where each sequence predicts the next price, then converted into PyTorch tensors and loaded in batches of 64.

### Step 2: Model Definition and Setup
A two-layer RNN model is built with hidden size 64 and dropout of 0.2, followed by a fully connected layer to output a single price value. The model has 12,673 trainable parameters, uses MSE Loss as the criterion and Adam optimizer with a learning rate of 0.001.

### Step 3: Training and Evaluation
The model is trained for 50 epochs with backpropagation updating weights each batch. Training loss is recorded and plotted per epoch. After training, predictions on the test set are inverse transformed to original price scale and compared against actual prices.

## Program
#### Name: Ramitha Chowdary S
#### Register Number: 212224240130

```Python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load and Preprocess Data
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary
# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

# Plot Training Loss
print('Name: Ramitha Chowdary S')
print('Register Number: 212224240130')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: Ramitha Chowdary S')
print('Register Number: 212224240130')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```

## Output
### Model Architecture Summary
<img width="605" height="247" alt="image" src="https://github.com/user-attachments/assets/6722b55c-3984-4007-860b-de76c2d8c02f" />

### Training Loss Over Epochs
<img width="502" height="415" alt="image" src="https://github.com/user-attachments/assets/743fe70b-13e1-4a71-bf87-337bf020bf9a" />


### Stock Price Prediction using RNN
<img width="719" height="521" alt="image" src="https://github.com/user-attachments/assets/c73487ea-d7f9-43f2-89a0-4c1173babd44" />

## Result
The model converged well with training loss dropping from 0.00083 to 0.00041 over 50 epochs. The final predicted price was 1099.71 against the actual price of 1115.65, showing close approximation. The prediction plot confirms the model captures the overall stock price trend with a minor lag at sharp turning points.


