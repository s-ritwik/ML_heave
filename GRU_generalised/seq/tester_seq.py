import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Define the Custom GRU Model (same as training)
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

# Inference parameters (match training configuration)
input_size = 1
hidden_sizes = [256, 128, 64]
output_size = 1  # Predict one data point at each time step
output_seq_len = 120  # Predict for 120 steps (6 seconds at 20Hz)
sequence_length = 800  # Initial sequence length
sample_rate_hz = 20  # Assume data is at 20 Hz (20 samples per second)

# Define the time steps for error calculation (2, 3, and 5 seconds)
steps_2s = 2 * sample_rate_hz  # 2 seconds of data
steps_3s = 3 * sample_rate_hz  # 3 seconds of data
steps_5s = 5 * sample_rate_hz  # 5 seconds of data

# Device configuration (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = CustomGRUModel(input_size, hidden_sizes, output_size).to(device)
model.load_state_dict(torch.load('D1Gru_40_6_256_128_64_seq.pth', map_location=device))
model.eval()

# Load the test data
test_file_path = 'train_data_normalised/D1H3_normalised.csv'  # Adjust the path
test_data = pd.read_csv(test_file_path, header=None).iloc[:, 0].values

# Conversion factor (meters to centimeters)
meters_to_cm = 25  # Convert meters to centimeters

# Duration of the test in seconds
x_seconds = 60  # Define how long you want the test to run
num_test_steps = int(x_seconds * sample_rate_hz)  # Number of test steps for x seconds

# Initialize hidden states
h_states = [
    torch.zeros(1, hidden_size).to(device)
    for hidden_size in hidden_sizes
]

# Prepare initial input sequence
real_time_input = test_data[:sequence_length]

# Variables to track time and errors
prediction_times = []
absolute_errors = []

# Errors for 2s, 3s, 5s
errors_2s = []
errors_3s = []
errors_5s = []

# Real-time simulation
plt.figure(figsize=(12, 8))
plt.ion()  # Enable interactive mode for live plotting

with torch.no_grad():
    for i in range(sequence_length, sequence_length + num_test_steps):
        start_time_loop = time.time()
        
        # Get the new data point
        input_t = torch.tensor([[test_data[i]]], dtype=torch.float32).to(device)  # Shape: (1, 1)
        
        # Forward pass
        output, h_states = model(input_t, h_states)
        
        # Predict future steps
        pred_h_states = [h_state.clone() for h_state in h_states]
        input_pred = output
        predictions = []
        for _ in range(output_seq_len):
            pred_output, pred_h_states = model(input_pred, pred_h_states)
            predictions.append(pred_output)
            input_pred = pred_output
        
        predicted_future = torch.cat(predictions).cpu().numpy().flatten()
        
        end_time = time.time()
        prediction_time = end_time - start_time_loop
        prediction_times.append(prediction_time)
        
        # Calculate the absolute error (in cm)
        true_future = test_data[i+1:i+1 + output_seq_len] * meters_to_cm
        predicted_future_cm = predicted_future * meters_to_cm
        abs_error = np.abs(true_future - predicted_future_cm)
        absolute_errors.append(abs_error.mean())
        
        # Calculate errors for the first 2, 3, and 5 seconds
        if len(true_future) >= steps_2s:
            errors_2s.append(np.mean(abs_error[:steps_2s]))  # Error for the first 2 seconds
        if len(true_future) >= steps_3s:
            errors_3s.append(np.mean(abs_error[:steps_3s]))  # Error for the first 3 seconds
        if len(true_future) >= steps_5s:
            errors_5s.append(np.mean(abs_error[:steps_5s]))  # Error for the first 5 seconds
        
        # Update hidden states
        h_states = [h_state.detach() for h_state in h_states]
        
        # Plotting
        plt.clf()
        plt.subplot(2, 1, 1)
        real_time_input_cm = test_data[i-sequence_length+1:i+1] * meters_to_cm
        plt.plot(range(i-sequence_length+1, i+1), real_time_input_cm, label='Input Data (cm)', color='b')
        plt.plot(range(i+1, i+1 + output_seq_len), true_future, 'g--', label='True Future Data (cm)')
        plt.plot(range(i+1, i+1 + output_seq_len), predicted_future_cm, 'r', label='Predicted Data (cm)')
        plt.text(i + output_seq_len, np.mean(predicted_future_cm), f"Time: {prediction_time:.4f} s", fontsize=9)
        plt.legend()
        
        # Plot the absolute error
        plt.subplot(2, 1, 2)
        plt.plot(range(i+1, i+1 + output_seq_len), abs_error, 'b', label='Absolute Error (cm)')
        plt.xlabel('Time Step')
        plt.ylabel('Error (cm)')
        plt.legend()
        
        plt.pause(0.01)
    
    plt.ioff()
    plt.show()
# Close all figures after the loop
plt.ioff()  # Turn off interactive mode
plt.close('all')  # Close all plots
# Calculate mean errors for 2, 3, and 5 seconds
avg_error_2s = np.mean(errors_2s)
avg_error_3s = np.mean(errors_3s)
avg_error_5s = np.mean(errors_5s)

# Calculate the overall average prediction time and error
avg_prediction_time = np.mean(prediction_times)
avg_absolute_error = np.mean(absolute_errors)

# Print the results
print(f"Average Prediction Time: {avg_prediction_time:.4f} seconds")
print(f"Average Absolute Error (2 seconds): {avg_error_2s:.4f} cm")
print(f"Average Absolute Error (3 seconds): {avg_error_3s:.4f} cm")
print(f"Average Absolute Error (5 seconds): {avg_error_5s:.4f} cm")
print(f"Total Average Absolute Error: {avg_absolute_error:.4f} cm")
