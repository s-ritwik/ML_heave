import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# __________________________________INPUTS__________________________________________________

sequence_length = 800  # Number of past time steps to use for prediction
output_size = 120      # Number of future time steps to predict
hidden_sizes = [1024, 1024]  # Hidden sizes for each GRU layer
batch_size = 128
epochs = 300       # Number of epochs for training
learning_rate = 0.001  # Initial learning rate

# ____________________________________________________________________________________#

# Specify the GPU index (0 or 1)
gpu_index = 1  # Set to 1 if you want to use the second GPU

# Set the environment variable to specify the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

# Set the device for GPU usage if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data directory (adjust as needed)
data_dir = 'train_data_normalised'
model_dir = 'GRU_models'
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
# scaler_input = MinMaxScaler(feature_range=(-1, 1))
# train_data = scaler_input.fit_transform(train_data.reshape(-1, 1)).flatten()

# Define GRU model with different hidden sizes per layer
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160):
        super(GRUModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.gru_layers = nn.ModuleList()
        # First GRU layer
        self.gru_layers.append(nn.GRU(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        # Additional GRU layers
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_sizes[i-1], hidden_sizes[i], num_layers=1, batch_first=True))
        # Fully connected layer
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x, h):
        h_out = []
        out = x
        for i, gru in enumerate(self.gru_layers):
            out, h_i = gru(out, h[i])
            h_out.append(h_i)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.tanh(out)
        return out, h_out
    
    def init_hidden(self, batch_size):
        # Initialize hidden states for all layers
        h = []
        for hidden_size in self.hidden_sizes:
            h.append(torch.zeros(1, batch_size, hidden_size).to(device))
        return h

# Initialize the model
print("Initializing the model...")
input_size = 1  # Since we're using univariate time series data
model = GRUModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler: Multiply the LR by 0.98 every epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)

print("Model initialized.")

# Prepare data for training (sequential sliding window)
X_train, y_train = [], []
for i in range(len(train_data) - sequence_length - output_size):
    X_train.append(train_data[i:i + sequence_length])
    y_train.append(train_data[i + sequence_length:i + sequence_length + output_size])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape data for GRU input
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Shape: (samples, sequence_length, input_size)
y_train = torch.tensor(y_train, dtype=torch.float32)  # Shape: (samples, output_size)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create directory for plots if it doesn't exist
plot_dir = 'GRU_plots'
os.makedirs(plot_dir, exist_ok=True)

# Timing the training process
start_time = time.time()

# Parameters for the weighted loss
x_seconds = 5  # First x seconds to apply the weight
w = 20  # Weight factor for the first x seconds

# Calculate how many time steps are in the first x seconds (20Hz data)
x_time_steps = x_seconds * 20  # Since it's 20Hz, multiply by 20

# Store loss values for plotting
all_losses = []
best_loss = float('inf')  # Track the best (minimum) loss

# Training Loop with Weighted Loss and Continuity Term
print("Starting training with weighted loss and continuity term...")
for epoch in range(epochs):
    running_loss = 0.0  # Track loss over the epoch
    total_loss = 0.0  # Accumulate loss for averaging
    model.train()  # Set model to training mode
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size_actual = inputs.size(0)
        # Initialize hidden states
        h = model.init_hidden(batch_size_actual)
        optimizer.zero_grad()
        
        # Forward pass
        outputs, h = model(inputs, h)
        
        # Separate the first x_time_steps and remaining time steps
        first_x_steps = targets[:, :x_time_steps]  # First x seconds
        remaining_steps = targets[:, x_time_steps:]  # Remaining steps after x seconds
        
        # Compute the loss for the first x seconds with weight w
        loss_first_x_steps = criterion(outputs[:, :x_time_steps], first_x_steps) * w
        
        # Compute the loss for the remaining steps without any weight
        if remaining_steps.size(1) > 0:
            loss_remaining_steps = criterion(outputs[:, x_time_steps:], remaining_steps)
        else:
            loss_remaining_steps = 0
        
        # Continuity loss: penalizing large differences between consecutive predictions
        continuity_loss = torch.mean((outputs[:, 1:] - outputs[:, :-1]) ** 2)
        
        # Total weighted loss including continuity term
        loss = loss_first_x_steps + loss_remaining_steps + 0.05 * continuity_loss  # Adjust 0.05 factor as needed
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 1000 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {avg_loss:.8f}")
    
    # Calculate average loss for the epoch
    avg_epoch_loss = total_loss / len(train_loader)
    all_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.8f}")
    
    # Save the model if the current loss is lower than the best loss
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        hidden_sizes_str = "_".join(map(str, hidden_sizes))
        model_name = f"D1_GRU_{sequence_length//20}_{output_size//20}_{hidden_sizes_str}.pth"
        model_path = os.path.join(model_dir, model_name)
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved at {model_path} with loss: {best_loss:.8f}")
    
    # Step the learning rate scheduler (multiply by 0.98 each         model_name = f"D1_GRU_{sequence_length//20}_{output_size//20}_{hidden_sizes_str}.pth"epoch)
    scheduler.step()

# Training time
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")
model_name = f"D1_GRU_{sequence_length//20}_{output_size//20}_{hidden_sizes_str}.pth"
# Plot and save error vs epoch
plt.figure()
plt.plot(range(1, epochs + 1), all_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Error vs. Epoch')
plt.savefig(os.path.join(plot_dir, 'error_vs_epoch_{model_name}.png'))
plt.close()
