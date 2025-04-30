import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import os
from sklearn.preprocessing import MinMaxScaler
from matplotlib.animation import FFMpegWriter  # Import FFMpegWriter


# __________________________________INPUTS__________________________________________________

test_time = 150
model_path = 'GRU_models_seq/D1_new_GRU_40_6_4048_4048.pth'  # Update with the correct path

# __________________________________________________________________________________________

# Function to extract parameters from model filename
def parse_model_path(model_path):
    # Find the substring starting with "D1_GRU"
    start_idx = model_path.find("_GRU_")
    if start_idx == -1:
        raise ValueError("Model path does not contain 'D1_GRU'.")
    
    # Extract the relevant part of the path
    filename = model_path[start_idx:]
    pattern = r"_GRU_(\d+)_(\d+)_(\d+(?:_\d+)*)\.pth"
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
os.makedirs("predictions", exist_ok=True)

# Format the new filename
hidden_sizes_str = "_".join(map(str, hidden_sizes))
new_filename = f"{sequence_length//20}_{output_size//20}_{hidden_sizes_str}.mp4"

# Full path for saving the file
new_filepath = os.path.join("predictions", new_filename)

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

# If test_data is already normalized, you can skip normalization
# If not, ensure to normalize it using the same scaler as used during training

# Conversion factor (meters to centimeters)
meters_to_cm = 25

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

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Set up formatting for the movie files
metadata = dict(title='Model Predictions', artist='Matplotlib', comment='GRU Model Predictions')
writer = FFMpegWriter(fps=sampling_rate, metadata=metadata)
folder_path = "seq/seq_results"
os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
# Start the writer
with writer.saving(fig, new_filepath, dpi=300):
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
                # Clear the axes
                ax1.clear()
                ax2.clear()

                # Plotting the data on ax1
                ax1.plot(range(i - sequence_length, i), test_data[i - sequence_length:i] * meters_to_cm, label='Input Data (cm)')
                ax1.plot(range(i + 1, i + 1 + output_size), true_future, 'g--', label='True Future Data (cm)')
                ax1.plot(range(i + 1, i + 1 + output_size), predicted_future, 'r', label='Predicted Data (cm)')
                # Add the time elapsed over total time as a suptitle
                fig.suptitle(f"Time Elapsed: {total_time_elapsed:.2f} s / {test_time} s", fontsize=12)
                ax1.legend()

                # Plotting the absolute error on ax2
                ax2.plot(range(i + 1, i + 1 + output_size), abs_error, 'b', label='Absolute Error (cm)')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Error (cm)')
                ax2.legend()

                # Grab the frame
                writer.grab_frame()

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

# Close the figure
plt.close(fig)

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
