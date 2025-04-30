import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import os
# Define the Temporal Block for TCN
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolution
        out = self.conv1(x)
        # Remove extra padding to match input length
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second convolution
        out = self.conv2(out)
        # Remove extra padding to match input length
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# Define the TCN model
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_size=160):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]  # Take the last time step
        out = self.fc(out)
        out = self.tanh(out)
        return out

# Function to extract parameters from model filename
def parse_model_path(model_path):
    filename = os.path.basename(model_path)
    pattern = r"D1_TCN_(\d+)_(\d+)_(\d+(?:_\d+)*)\.pth"
    match = re.match(pattern, filename)
    if match:
        sequence_length = int(match.group(1)) * 20  # Convert from seconds to points at 20Hz
        output_size = int(match.group(2)) * 20  # Convert from seconds to points at 20Hz
        num_channels = list(map(int, match.group(3).split("_")))  # List of hidden layers
        return sequence_length, output_size, num_channels
    else:
        raise ValueError("Model path does not match expected pattern.")

# Set the model path
model_path = 'D1_TCN_40_8_512_512.pth'  # Example path, replace with actual path

# Parse model parameters
sequence_length, output_size, num_channels = parse_model_path(model_path)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = TemporalConvNet(num_inputs=1, num_channels=num_channels, output_size=output_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the test data from the CSV file
test_file_path = 'train_data_normalised/D1H3_normalised.csv'  # Update with the correct path if necessary
test_data = pd.read_csv(test_file_path, header=None).iloc[:, 0].values

# Define the starting and ending indices
start_index = 20 * 1  # Starting index for data
end_index = 20 * 60 + sequence_length  # Ending index, corresponding to 30 seconds of data at 20Hz

# Extract only the desired segment of the data
test_data = test_data[start_index:end_index]

# Conversion factor (meters to centimeters)
meters_to_cm = 25  # Convert meters to centimeters

# Real-time simulation
plt.figure(figsize=(12, 8))
plt.ion()  # Turn on interactive mode for live plotting

# Variables to track time and error
prediction_times = []
absolute_errors = []

# Errors for the first 3, 4, and 5 seconds
errors_3s = []
errors_4s = []
errors_5s = []

# Define the number of steps for 3, 4, and 5 seconds
steps_3s = 3 * 20  # 3 seconds at 20Hz
steps_4s = 4 * 20  # 4 seconds at 20Hz
steps_5s = 5 * 20  # 5 seconds at 20Hz

# Simple Moving Average Filter
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

with torch.no_grad():
    real_time_input = test_data[:sequence_length]
    
    for i in range(sequence_length, len(test_data) - output_size):
        start_time_loop = time.time()
        
        # Prepare input tensor for TCN (batch_size=1, channels=1, sequence_length)
        input_tensor = torch.tensor(real_time_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Measure prediction time
        start_time = time.time()
        predicted = model(input_tensor).cpu().numpy().flatten()
        
        # Apply the low-pass filter (SMA)
        predicted_filtered = moving_average(predicted, window_size=1)
        predicted_filtered = np.concatenate((predicted[:1 - 1], predicted_filtered))
        end_time = time.time()

        # Calculate the time taken for prediction
        prediction_time = end_time - start_time
        prediction_times.append(prediction_time)

        # Calculate the absolute error (in cm)
        true_future = test_data[i:i + output_size] * meters_to_cm
        predicted_future = predicted_filtered * meters_to_cm
        abs_error = np.abs(true_future - predicted_future)
        absolute_errors.append(abs_error.mean())

        # Calculate errors for the first 3, 4, and 5 seconds
        errors_3s.append(np.mean(abs_error[:steps_3s]))
        errors_4s.append(np.mean(abs_error[:steps_4s]))
        errors_5s.append(np.mean(abs_error[:steps_5s]))

        # Shift the window and append the new data
        real_time_input = np.append(real_time_input[1:], test_data[i])

        # Clear the previous plot
        plt.clf()

        # Plotting the data
        plt.subplot(2, 1, 1)
        plt.plot(range(i - sequence_length, i), real_time_input * meters_to_cm, label='Input Data (cm)')
        plt.plot(range(i, i + output_size), true_future, 'g--', label='True Future Data (cm)')
        plt.plot(range(i, i + output_size), predicted_future, 'r', label='Predicted Data (cm)')
        plt.text(i + output_size, np.mean(predicted_future), f"Time: {prediction_time:.4f} s", fontsize=9)
        plt.legend()

        # Plotting the absolute error
        plt.subplot(2, 1, 2)
        plt.plot(range(i, i + output_size), abs_error, 'b', label='Absolute Error (cm)')
        plt.xlabel('Time Step')
        plt.ylabel('Error (cm)')
        plt.legend()

        plt.pause(0.01)  # Pause to update the plot
        end_time_loop = time.time()
        prediction_time_loop = end_time_loop - start_time_loop

# Close all figures after the loop
plt.ioff()
plt.close('all')

# Calculate average errors and prediction time
avg_prediction_time = np.mean(prediction_times)
avg_absolute_error = np.mean(absolute_errors)
avg_error_3s = np.mean(errors_3s)
avg_error_4s = np.mean(errors_4s)
avg_error_5s = np.mean(errors_5s)

print(f"Average Prediction Time: {avg_prediction_time:.4f} seconds")
print(f"Average Absolute Error for the first 3 seconds: {avg_error_3s:.4f} cm")
print(f"Average Absolute Error for the first 4 seconds: {avg_error_4s:.4f} cm")
print(f"Average Absolute Error for the first 5 seconds: {avg_error_5s:.4f} cm")
print(f"Total Average Absolute Error: {avg_absolute_error:.4f} cm")
