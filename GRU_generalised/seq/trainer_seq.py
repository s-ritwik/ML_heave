import torch
import torch.nn as nn
import torch.optim as optim
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
    return data

# Get the list of CSV files
csv_files = os.listdir(data_dir)

# Filter files for training to include all D1H files except those containing 'D1H3'
train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]

# Load and normalize training data from the filtered D1H files
train_data = load_data(train_files)
train_data = normalize_data(train_data)
print(f"Loaded and normalized training data from {len(train_files)} files.")

# Training parameters
input_size = 1
hidden_sizes = [256, 128, 64]
output_size = 1  # Predict one data point at each time step
output_seq_len = 120  # Predict the next 120 data points
sequence_length = 800  # Context window for backpropagation
batch_size = 128
learning_rate = 0.001
epochs = 100
tbptt_step = 20  # Truncated backpropagation through time step size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare sequences
# Sliding window approach to ensure every data point is used
inputs = []
targets = []
for i in range(len(train_data) - sequence_length - output_seq_len):
    input_seq = train_data[i:i + sequence_length]
    target_seq = train_data[i + sequence_length:i + sequence_length + output_seq_len]
    inputs.append(input_seq)
    targets.append(target_seq)

# Convert to NumPy arrays
inputs = np.array(inputs)
targets = np.array(targets)

# Reshape into batches
num_batches = len(inputs) // batch_size
inputs = inputs[:num_batches * batch_size]
targets = targets[:num_batches * batch_size]

# Convert to tensors and add feature dimension
inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_samples, sequence_length, 1)
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_samples, output_seq_len, 1)

# Create TensorDataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the Custom GRU Model with GRUCell
class CustomGRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomGRUModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        # Initialize GRU cells
        self.gru_cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            self.gru_cells.append(nn.GRUCell(in_size, hidden_sizes[i]))
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, input_t, h_states):
        new_h_states = []
        out = input_t
        for i, gru_cell in enumerate(self.gru_cells):
            h_state = gru_cell(out, h_states[i])
            out = h_state
            new_h_states.append(h_state)
        output = self.fc(out)
        output = torch.tanh(output)
        return output, new_h_states

# Initialize the model
print("Initializing the model...")
model = CustomGRUModel(input_size, hidden_sizes, output_size).to(device)

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRUCell):
        nn.init.xavier_uniform_(m.weight_ih)
        nn.init.xavier_uniform_(m.weight_hh)
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

model.apply(initialize_weights)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the scheduler to decrease the learning rate by 0.98 every epoch
gamma = 0.98
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Training loop with TBPTT
print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0.0
    for inputs_batch, targets_batch in train_loader:
        inputs_batch = inputs_batch.to(device)
        targets_batch = targets_batch.to(device)
        batch_size_curr = inputs_batch.size(0)
        
        # Initialize hidden states
        h_states = [torch.zeros(batch_size_curr, hidden_size).to(device) for hidden_size in hidden_sizes]
        
        optimizer.zero_grad()
        
        for t in range(0, sequence_length, tbptt_step):
            inputs_chunk = inputs_batch[:, t:t+tbptt_step, :]
            
            for step in range(inputs_chunk.size(1)):
                input_t = inputs_chunk[:, step, :]
                
                # Forward pass
                output, h_states = model(input_t, h_states)
                
            # Detach hidden states after every step to prevent backpropagation through the entire history
            h_states = [h_state.detach() for h_state in h_states]
        
        # After processing the input sequence, predict future steps
        pred_h_states = [h_state.clone() for h_state in h_states]
        input_pred = output
        predictions = []
        for _ in range(output_seq_len):
            pred_output, pred_h_states = model(input_pred, pred_h_states)
            predictions.append(pred_output.unsqueeze(1))
            input_pred = pred_output
        
        predictions = torch.cat(predictions, dim=1)
        
        # Compute loss
        loss = criterion(predictions, targets_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, Learning Rate: {current_lr:.8f}")
    scheduler.step()

print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# Save the trained model
model_save_path = 'D1Gru_40_6_256_128_64_seq.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}.")
