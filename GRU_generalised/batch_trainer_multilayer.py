import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import time

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

# Normalize the data
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

# Get the list of CSV files
csv_files = os.listdir(data_dir)

# Filter files for training to include all D1H files except those containing 'D1H3'
train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
test_files = [file for file in csv_files if 'D1H3' in file]

# Load and normalize training data from the filtered D1H files
train_data = load_data(train_files)
train_data = normalize_data(train_data)
print(f"Loaded and normalized training data from {len(train_files)} files.")

# Training parameters
input_size = 1
hidden_sizes = [ 512,128,64]
output_size = 120  # Predict next 120 steps
learning_rate = 0.001
epochs = 1000
sequence_length = 800  # Training context window
batch_size = 256 # Adjusted for batch processing

# Create sequences of inputs and targets
print("Preparing dataset...")
inputs = []
targets = []
step_size = 100  # Adjust step_size as needed to control data size

for i in range(0, len(train_data) - sequence_length - output_size, step_size):
    input_seq = train_data[i:i+sequence_length]
    target_seq = train_data[i+sequence_length:i+sequence_length+output_size]
    inputs.append(input_seq)
    targets.append(target_seq)

# Convert lists to numpy arrays
inputs = np.array(inputs)
targets = np.array(targets)

# Convert numpy arrays to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_samples, sequence_length, 1)
targets = torch.tensor(targets, dtype=torch.float32)  # Shape: (num_samples, output_size)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the Custom GRU Model with Separate GRU Layers
class CustomGRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomGRUModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(input_size, hidden_sizes[0], batch_first=True))
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_sizes[i - 1], hidden_sizes[i], batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x, h_states):
        out = x
        new_h_states = []
        for i, gru in enumerate(self.gru_layers):
            out, h_state = gru(out, h_states[i])
            new_h_states.append(h_state)
        out = self.fc(out[:, -1, :])
        out = torch.tanh(out)
        return out, new_h_states

# Initialize the model
print("Initializing the model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomGRUModel(input_size, hidden_sizes, output_size).to(device)

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

model.apply(initialize_weights)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the scheduler to decrease the learning rate at each epoch
gamma = 0.97  # Factor by which the learning rate will be reduced
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Training Loop with Batch Processing
print("Starting batch training...")
start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs_batch, targets_batch) in enumerate(train_loader):
        inputs_batch = inputs_batch.to(device)  # Shape: (batch_size, sequence_length, input_size)
        targets_batch = targets_batch.to(device)  # Shape: (batch_size, output_size)
        batch_size_curr = inputs_batch.size(0)
        
        # Initialize hidden states for each batch
        h_states = [
            torch.zeros(1, batch_size_curr, hidden_size).to(device)
            for hidden_size in hidden_sizes
        ]
        
        optimizer.zero_grad()
        
        # Forward pass
        output, h_states = model(inputs_batch, h_states)
        
        # Detach hidden states to prevent backpropagating through the entire history
        h_states = [h_state.detach() for h_state in h_states]
        
        # Calculate loss
        loss = criterion(output, targets_batch)
        
        # Detect NaN loss and skip if found
        if torch.isnan(loss):
            print(f"NaN detected at batch {i}, skipping this batch.")
            continue
        
        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Print progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Batch {i+1}/{len(train_loader)} "
                  f"Loss: {loss.item():.8f}")
    
    # Step the scheduler at the end of each epoch
    scheduler.step()
    
    # Print average loss and current learning rate after each epoch
    avg_epoch_loss = running_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.8f}, Learning Rate: {current_lr:.8f}")

# Calculate and print total training time
total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds.")

# Save the trained model
model_save_path = 'D1Gru_40_6_256_128_64_batch_trained.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}.")
