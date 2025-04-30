
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Specify the GPU index (0 or 1)
gpu_index = 1  # Set to 1 if you want to use the second GPU

# Set the environment variable to specify the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
# Set the directory for the CSV files
data_dir = 'train_data_normalised'
print("Starting data loading and preprocessing...")

# Function to load data from CSV files
def load_data(file_list):
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data)

# Get the list of CSV files
csv_files = os.listdir(data_dir)

# Filter files for training to include all D1H files except those containing 'D1H3_75'
train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
test_files = [file for file in csv_files if 'D1H3' in file]

# Load training data from the filtered D1H files
train_data = load_data(train_files)
print(f"Loaded training data from {len(train_files)} files.")

# Prepare training sequences
input_size = 1
hidden_sizes = [512,512]  # Define hidden sizes for each GRU layer
output_size = 140  # Predict next 20x steps (x seconds at 20Hz)
num_layers = len(hidden_sizes)
learning_rate = 0.001
epochs = 1000
sequence_length = 800  # 40 seconds of data (20Hz)
window_size_avg = 7  # For moving average filter

X_train, y_train = [], []

print("Preparing training sequences...")
for i in range(len(train_data) - sequence_length - output_size):
    X_train.append(train_data[i:i + sequence_length])
    y_train.append(train_data[i + sequence_length:i + sequence_length + output_size])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize the input and output separately
#scaler_input = MinMaxScaler(feature_range=(-1, 1))
#X_train = scaler_input.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

#scaler_target = MinMaxScaler(feature_range=(-1, 1))
#y_train = scaler_target.fit_transform(y_train)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Shape: (samples, sequence_length, 1)
y_train = torch.tensor(y_train, dtype=torch.float32)  # Shape: (samples, output_size)

print(f"Total training samples: {X_train.shape[0]}")

# Create DataLoader for training
batch_size = 256
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("DataLoader created.")

# Define the Custom GRU Model with Dropout
class CustomGRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomGRUModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        # Initialize ModuleList to hold GRU layers
        self.gru_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.2)  # Add dropout with 20% probability
        
        # First GRU layer
        self.gru_layers.append(nn.GRU(input_size, hidden_sizes[0], batch_first=True))
        
        # Additional GRU layers
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
        
        # Fully connected layer for the output
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        out = x  # Initial input
        
        # Initialize hidden states for each GRU layer
        h_states = [
            torch.zeros(1, x.size(0), hidden_size).to(x.device)
            for hidden_size in self.hidden_sizes
        ]
        
        # Pass through each GRU layer sequentially
        for i, gru in enumerate(self.gru_layers):
            out, h_states[i] = gru(out, h_states[i])
            out = self.dropout(out)  # Apply dropout after each GRU layer
        
        # Use the output of the last GRU layer at the final time step
        out = self.fc(out[:, -1, :])  # Shape: (batch_size, output_size)
        out = torch.tanh(out)  # Apply tanh activation to constrain output
        return out

# Initialize the model
print("Initializing the model...")
model = CustomGRUModel(input_size, hidden_sizes, output_size).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler: Multiply the LR by 0.98 every epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)

print("Model initialized.")

# Timing the training process
start_time = time.time()

# Parameters for the weighted loss
x_seconds = 5  # First x seconds to apply the weight
w = 30  # Weight factor for the first x seconds

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
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Separate the first x_time_steps and remaining time steps
        first_x_steps = targets[:, :x_time_steps]  # First x seconds
        remaining_steps = targets[:, x_time_steps:]  # Remaining steps after x seconds
        
        # Compute the loss for the first x seconds with weight w
        loss_first_x_steps = criterion(outputs[:, :x_time_steps], first_x_steps) * w
        
        # Compute the loss for the remaining steps without any weight
        loss_remaining_steps = criterion(outputs[:, x_time_steps:], remaining_steps)
        
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
        if (batch_idx + 1) % 100 == 0:
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
        torch.save(model.state_dict(), 'D1Gru_40_7_512_512_fullytrained.pth')
        print(f"New best model saved with loss: {best_loss:.8f}")
    
    # Step the learning rate scheduler (multiply by 0.98 each epoch)
    scheduler.step()

# Calculate and print the total training time
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

# Plot the training loss
plt.plot(all_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()
