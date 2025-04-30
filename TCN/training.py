import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ExponentialLR  # Added import

#__________________________________INPUTS__________________________________________________

sequence_length = 800  # Last 800 data points used for prediction
output_size = 160  # Predict next 160 steps
num_channels = [1024, 1024]
batch_size = 1024
epochs = 100  # Increased number of epochs

#____________________________________________________________________________________#
# Specify the GPU index (0 or 1)
gpu_index = 0  # Set to 1 if you want to use the second GPU

# Set the environment variable to specify the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

# Set the device for GPU usage if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data directory (adjust as needed)
data_dir = 'train_data_normalised'
model_dir = 'TCN_models'
os.makedirs(model_dir, exist_ok=True)  # Create the model directory if it doesn't exist

# Load and preprocess data
def load_data(file_list):
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data)

# Get CSV files
csv_files = os.listdir(data_dir)
train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
test_files = [file for file in csv_files if 'D1H3' in file]

# Load training data
train_data = load_data(train_files)

# Normalize the data
scaler_input = MinMaxScaler(feature_range=(-1, 1))
train_data = scaler_input.fit_transform(train_data.reshape(-1, 1)).flatten()

# Define TCN model
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Calculate padding for causal convolution
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply causal padding to ensure the output length matches the input length
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Remove extra padding from the right side
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]  # Remove extra padding from the right side
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Ensure the residual connection matches the shape of out
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_size=160):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        out = self.network(x)
        out = out[:, :, -1]  # Take the last time step
        out = self.fc(out)
        out = self.tanh(out)
        return out

# Initialize the TCN model
model = TemporalConvNet(num_inputs=1, num_channels=num_channels, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize the learning rate scheduler
scheduler = ExponentialLR(optimizer, gamma=0.97)  # Learning rate decay factor of 0.97

# Prepare data for training (sequential sliding window)
X_train, y_train = [], []
for i in range(len(train_data) - sequence_length - output_size):
    X_train.append(train_data[i:i + sequence_length])
    y_train.append(train_data[i + sequence_length:i + sequence_length + output_size])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape data for TCN input
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Shape: (samples, channels, sequence_length)
y_train = torch.tensor(y_train, dtype=torch.float32)  # Shape: (samples, output_size)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create directory for plots if it doesn't exist
plot_dir = 'TCN_plots'
os.makedirs(plot_dir, exist_ok=True)

# Track average losses per epoch
avg_losses = []

# Training the TCN model with best model saving
best_loss = float('inf')
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate and store average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    avg_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # Update the learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Current learning rate: {current_lr:.6f}")

    # Check if this is the best model so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        # Construct model save path using the specified naming convention
        model_name = f"D1_TCN_{sequence_length//20}_{output_size//20}_" + "_".join(map(str, num_channels)) + ".pth"
        model_path = os.path.join(model_dir, model_name)
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved at {model_path} with loss {best_loss:.6f}")

print(f"Model training completed and saved at {model_path}.")

# Plot and save error vs epoch
plt.figure()
plt.plot(range(1, epochs + 1), avg_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Error vs. Epoch')
plt.savefig(os.path.join(plot_dir, 'error_vs_epoch.png'))
plt.close()
