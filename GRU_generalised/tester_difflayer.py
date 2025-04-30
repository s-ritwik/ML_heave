import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Define the Custom GRU Model to match the training code
class CustomGRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomGRUModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        # Initialize ModuleList to hold GRU layers
        self.gru_layers = nn.ModuleList()
        
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

        # Use the output of the last GRU layer at the final time step
        out = self.fc(out[:, -1, :])  # Shape: (batch_size, output_size)
        out = torch.tanh(out)  # Apply tanh activation
        return out

# Parameters
input_size = 1
hidden_sizes = [512,512]
output_size = 120  # Predict next 10 seconds (10 seconds at 20Hz)
sequence_length = 1000  # 50 seconds of data (20Hz)
window_size_avg = 1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = CustomGRUModel(input_size, hidden_sizes, output_size).to(device)
model.load_state_dict(torch.load('brute_models/noisy_D1Gru_50_6_512_512.pth', map_location=device))
model.eval()

# Load the test data from the CSV file
test_file_path = 'real_data/output_20Hz.csv'
test_data = pd.read_csv(test_file_path, header=None).iloc[:, 0].values

# Define the starting and ending indices
start_index = 20 * 1
end_index = 20 * 180 + sequence_length
test_data = test_data[start_index:end_index]

# Conversion factor (meters to centimeters)
meters_to_cm = 25

# Real-time simulation setup
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
plt.ion()  # Enable interactive mode
plt.show(block=False)  # Ensure the plot doesn't block the rest of the code

# Variables to track time and error
prediction_times = []
absolute_errors = []
errors_3s, errors_4s, errors_5s = [], [], []
steps_3s, steps_4s, steps_5s = 3 * 20, 4 * 20, 5 * 20

# Simple Moving Average Filter
def moving_average(data, window_size=5):
    return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Add noise to the test data
noise_level = 0  # Define the noise level
sigma = np.std(test_data) * noise_level
noise = np.random.normal(0.0, sigma, test_data.shape)
test_data_noisy = test_data + noise  # Noisy test data

with torch.no_grad():
    real_time_input = test_data_noisy[:sequence_length]  # Use noisy data
    
    for i in range(sequence_length, len(test_data_noisy) - output_size):
        start_time_loop = time.time()
        
        # Prepare input tensor
        input_tensor = torch.tensor(real_time_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        # Measure prediction time
        start_time = time.time()
        predicted = model(input_tensor).cpu().numpy().flatten()
        
        # Apply the low-pass filter (SMA)
        predicted_filtered = moving_average(predicted, window_size=window_size_avg)
        predicted_filtered = np.concatenate((predicted[:window_size_avg - 1], predicted_filtered))
        end_time = time.time()
        prediction_time = end_time - start_time

        # Calculate the absolute error (in cm)
        true_future = test_data[i:i + output_size] * meters_to_cm
        predicted_future = predicted_filtered * meters_to_cm
        abs_error = np.abs(true_future - predicted_future)
        absolute_errors.append(abs_error.mean())

        # Calculate errors for the first 3, 4, and 5 seconds
        errors_3s.append(np.mean(abs_error[:steps_3s]))
        errors_4s.append(np.mean(abs_error[:steps_4s]))
        errors_5s.append(np.mean(abs_error[:steps_5s]))

        # Shift the window and append the new noisy data
        real_time_input = np.append(real_time_input[1:], test_data_noisy[i])

        # Clear the previous axes without closing the figure
        axs[0].cla()
        axs[1].cla()

        # Plot the input, true future, and predicted future data
        axs[0].plot(range(i - sequence_length, i), real_time_input * meters_to_cm, label='Input Data (cm)')
        axs[0].plot(range(i, i + output_size), true_future, 'g--', label='True Future Data (cm)')
        axs[0].plot(range(i, i + output_size), predicted_future, 'r', label='Predicted Data (cm)')
        axs[0].text(i + output_size, np.mean(predicted_future), f"Time: {prediction_time:.4f} s", fontsize=9)
        axs[0].legend()
        axs[0].set_title('Input, True Future, and Predicted Data')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Altitude (cm)')
        axs[0].grid(True)

        # Plot the absolute error
        axs[1].plot(range(i, i + output_size), abs_error, 'b', label='Absolute Error (cm)')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Error (cm)')
        axs[1].legend()
        axs[1].grid(True)

        # Update the figure canvas
        fig.canvas.draw()
        fig.canvas.flush_events()

        end_time_loop = time.time()
        prediction_time_loop = end_time_loop - start_time_loop

# After the loop is done, turn off interactive mode and close plots if desired
plt.ioff()
plt.close('all')

# Calculate the average prediction time and absolute errors
avg_prediction_time = np.mean(prediction_times)
avg_absolute_error = np.mean(absolute_errors)

# Calculate average errors for the first 3, 4, and 5 seconds
avg_error_3s = np.mean(errors_3s)
avg_error_4s = np.mean(errors_4s)
avg_error_5s = np.mean(errors_5s)

print(f"Average Prediction Time: {avg_prediction_time:.4f} seconds")
print(f"Average Absolute Error for the first 3 seconds: {avg_error_3s:.4f} cm")
print(f"Average Absolute Error for the first 4 seconds: {avg_error_4s:.4f} cm")
print(f"Average Absolute Error for the first 5 seconds: {avg_error_5s:.4f} cm")
print(f"Total Average Absolute Error: {avg_absolute_error:.4f} cm")
