import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import os
from sklearn.preprocessing import MinMaxScaler

# __________________________________INPUTS__________________________________________________

test_time = 80
model_path = 'seq/D1_GRU_40_8_1024_1024.pth'  # Update with the correct path
plot_option = 1  # Set to 0 for full plot with sequence length, set to 1 for only predicted and upcoming data

# __________________________________________________________________________________________

# Function to extract parameters from model filename
def parse_model_path(model_path):
    filename = os.path.basename(model_path)
    pattern = r"D1_GRU_(\d+)_(\d+)_(\d+(?:_\d+)*)\.pth"
    match = re.match(pattern, filename)
    if match:
        sequence_length = int(match.group(1)) * 20  # Convert from seconds to points at 20Hz
        output_size = int(match.group(2)) * 20      # Convert from seconds to points at 20Hz
        hidden_sizes = list(map(int, match.group(3).split("_")))
        return sequence_length, output_size, hidden_sizes
    else:
        raise ValueError("Model path does not match expected pattern.")

# Parse model parameters
sequence_length, output_size, hidden_sizes = parse_model_path(model_path)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the GRUModel class (same as in training script)
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

# Instantiate the model with parsed parameters
model = GRUModel(input_size=1, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the test data from the CSV file
test_file_path = 'train_data_normalised/D1H3_normalised.csv'  # Update with the correct path if necessary
test_data = pd.read_csv(test_file_path, header=None).iloc[:, 0].values

# Conversion factor (meters to centimeters)
meters_to_cm = 25

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

# Define the sampling rate
sampling_rate = 20  # Hz

# Convert test_time to total number of steps
total_steps = int(test_time * sampling_rate)  # Total steps equal to test_time seconds at 20Hz

# Define the starting index
start_index = max(sequence_length, output_size)

# Calculate the tentative ending index
end_index = start_index + total_steps

# Ensure indices stay within bounds of test_data
if end_index + output_size > len(test_data):
    # Adjust end_index to prevent out-of-bounds access
    end_index = len(test_data) - output_size
    # Recalculate total_steps based on the new end_index
    total_steps = end_index - start_index

print(f"Testing for {total_steps / sampling_rate} seconds.")
print(f"start_index: {start_index}")
print(f"end_index: {end_index}")
print(f"Length of test_data: {len(test_data)}")

desired_interval = 1.0 / sampling_rate  # This will be 0.05 seconds for 20 Hz

# Initialize hidden states
h = [torch.zeros(1, 1, hs).to(device) for hs in hidden_sizes]

# Record the start time of the testing loop
testing_start_time = time.time()

# Define error start index
error_start_index = start_index + sequence_length

# Testing loop
with torch.no_grad():
    for i in range(start_index, end_index):
        # Record the start time of the iteration
        iteration_start_time = time.time()

        # Prepare input tensor (batch_size=1, sequence_length=1, input_size=1)
        input_value = test_data[i]
        input_tensor = torch.tensor([[[input_value]]], dtype=torch.float32).to(device)

        # Measure prediction time
        model_start_time = time.time()
        output, h = model(input_tensor, h)

        # Detach hidden states to prevent backpropagation through time
        h = [h_i.detach() for h_i in h]

        # Convert output to numpy array
        predicted = output.cpu().numpy().flatten()

        # Calculate the time taken for prediction
        model_end_time = time.time()
        prediction_time = model_end_time - model_start_time
        prediction_times.append(prediction_time)

        # Calculate the total time elapsed since testing started
        total_time_elapsed = time.time() - testing_start_time

        # Generate time axis for plotting
        time_input = np.arange(i - sequence_length, i) * (1 / sampling_rate)
        time_future = np.arange(i + 1, i + 1 + output_size) * (1 / sampling_rate)

        # Calculate and record errors only after error_start_index
        if i >= error_start_index:
            # Calculate the absolute error (in cm)
            true_future = test_data[i + 1:i + 1 + output_size] * meters_to_cm
            predicted_future = predicted * meters_to_cm
            abs_error = np.abs(true_future - predicted_future)
            absolute_errors.append(abs_error.mean())

            # Calculate errors for the first 3, 4, and 5 seconds
            errors_3s.append(np.mean(abs_error[:steps_3s]))
            errors_4s.append(np.mean(abs_error[:steps_4s]))
            errors_5s.append(np.mean(abs_error[:steps_5s]))
        else:
            # Before error_start_index, set errors to zero or skip
            abs_error = np.zeros(output_size)
            predicted_future = predicted * meters_to_cm
            true_future = test_data[i + 1:i + 1 + output_size] * meters_to_cm

        # Plotting every time step
        if i % 1 == 0:
            # Clear the previous plot
            plt.clf()

            if plot_option == 0:
                # Plot with sequence length, true future data, predicted data, and error
                plt.subplot(2, 1, 1)
                plt.plot(time_input, test_data[i - sequence_length:i] * meters_to_cm, label='Input Data (cm)')
                plt.plot(time_future, true_future, 'g--', label='True Future Data (cm)')
                plt.plot(time_future, predicted_future, 'r', label='Predicted Data (cm)')
                plt.ylim(-30, 30)  # Set y-axis range for input and predicted data
                plt.suptitle(f"Time Elapsed: {total_time_elapsed:.2f} s / {test_time} s", fontsize=12)
                plt.xlabel('Time (seconds)')
                plt.legend()

            elif plot_option == 1:
                # Plot only predicted data, true future data, and error
                plt.subplot(2, 1, 1)
                plt.plot(time_future, true_future, 'g--', label='True Future Data (cm)')
                plt.plot(time_future, predicted_future, 'r', label='Predicted Data (cm)')
                plt.ylim(-30, 30)  # Set y-axis range for predicted data only
                plt.suptitle(f"Time Elapsed: {total_time_elapsed:.2f} s / {test_time} s", fontsize=12)
                plt.xlabel('Time (seconds)')
                plt.legend()

            # Plotting the absolute error (common for both options)
            plt.subplot(2, 1, 2)
            plt.plot(time_future, abs_error, 'b', label='Absolute Error (cm)')
            plt.ylim(0, 15)  # Set y-axis range for error
            plt.xlabel('Time (seconds)')
            plt.ylabel('Error (cm)')
            plt.legend()

            plt.pause(0.01)  # Pause to update the plot

        # Calculate the elapsed time for the iteration
        iteration_end_time = time.time()
        elapsed_time = iteration_end_time - iteration_start_time

        # Calculate the time to sleep to maintain the desired interval
        time_to_sleep = desired_interval - elapsed_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        else:
            # If processing took longer than desired_interval, no need to sleep
            pass

# Close all figures after the loop
plt.ioff()
plt.close('all')

# Calculate average errors and prediction time
if absolute_errors:
    avg_prediction_time = np.mean(prediction_times)
    avg_absolute_error = np.mean(absolute_errors)
    avg_error_3s = np.mean(errors_3s)
    avg_error_4s = np.mean(errors_4s)
    avg_error_5s = np.mean(errors_5s)

    # Print results
    print(f"Average Prediction Time: {avg_prediction_time:.4f} seconds")
    print(f"Average Absolute Error for the first 3 seconds: {avg_error_3s:.4f} cm")
    print(f"Average Absolute Error for the first 4 seconds: {avg_error_4s:.4f} cm")
    print(f"Average Absolute Error for the first 5 seconds: {avg_error_5s:.4f} cm")
    print(f"Total Average Absolute Error: {avg_absolute_error:.4f} cm")
else:
    print("No errors were recorded. Ensure that the test duration is sufficient and that the error_start_index is within the testing range.")
