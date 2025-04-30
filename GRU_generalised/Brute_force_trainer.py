import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
import ast
import matplotlib.pyplot as plt  # Make sure this is imported at the top of your script
from PIL import Image, ImageDraw, ImageFont  # Make sure to import these
import multiprocessing as mp
import matplotlib.pyplot as plt  # Make sure this is imported at the top of your script
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
# Function to load data from CSV files
def load_data(file_list, data_dir):
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data)

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




def save_plot_with_summary(all_losses, summary_line, model_name):
    # Create the folder path for storing the results
    folder_path = "brute_results"
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Create a figure for the error vs. epoch plot
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs. Epoch for {model_name}')
    plt.legend()
    plt.grid()

    # Save the plot to a temporary image file inside the folder
    temp_plot_filename = os.path.join(folder_path, f"{model_name}_temp_plot.png")
    plt.savefig(temp_plot_filename)
    plt.close()  # Close the plot to free up memory

    # Open the saved plot image
    plot_image = Image.open(temp_plot_filename)

    # Create a new blank image with extra space for the summary text
    new_width = plot_image.width
    new_height = plot_image.height + 200  # Extra space for text
    combined_image = Image.new("RGB", (new_width, new_height), "white")

    # Paste the plot image onto the new image
    combined_image.paste(plot_image, (0, 0))

    # Add the summary text below the plot
    draw = ImageDraw.Draw(combined_image)

    # Define the font (use a default font if PIL cannot find one)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Define text position
    text_position = (10, plot_image.height + 10)
    draw.text(text_position, summary_line, font=font, fill="black")

    # Save the combined image with a unique name inside the folder
    combined_filename = os.path.join(folder_path, f"{model_name}_error_vs_epoch_and_summary.png")
    combined_image.save(combined_filename)

    # Remove the temporary plot image
    os.remove(temp_plot_filename)

def train_and_test_model(config_line):
    # set_device(process_id, num_gpus)  # Set the GPU for the process
    # Create the folder path for storing the models
    torch.cuda.set_device(0)
    model_folder = "brute_models"
    os.makedirs(model_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Parse the configuration line
    config = {}
    for item in config_line.strip().split(';'):
        # Check if the item contains a colon to ensure it is a key-value pair
        if ':' not in item:
            print(f"Skipping malformed configuration item: {item}")
            continue  # Skip to the next item
        
        key, value = item.strip().split(':', 1)  # Split only at the first colon
        key = key.strip()
        value = value.strip()
        if key in ['hidden_sizes']:
            try:
                config[key] = ast.literal_eval(value)  # Parse list safely
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing {key}: {value} - {e}")
                return f"Error parsing {key}, skipping model"
        elif key in ['sequence_length', 'output_size', 'x_seconds', 'w']:
            try:
                config[key] = int(value)  # Convert value to integer
            except ValueError as e:
                print(f"Error converting {key} to int: {value} - {e}")
                return f"Error converting {key}, skipping model"
    
    # Check if all required parameters are present
    if not all(k in config for k in ['hidden_sizes', 'sequence_length', 'output_size', 'x_seconds', 'w']):
        print(f"Incomplete configuration, skipping: {config}")
        return f"Incomplete configuration, skipping: {config}"
    
    hidden_sizes = config['hidden_sizes']
    sequence_length = config['sequence_length']
    output_size = config['output_size']
    x_seconds = config['x_seconds']
    w = config['w']

    # Construct the model name with the new convention
    model_name = f"D1Gru_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}.pth"
    print(f"Sequence started for : {model_name} \n")

    # Load data
    data_dir = 'train_data_normalised'
    csv_files = os.listdir(data_dir)
    train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
    test_files = [file for file in csv_files if 'D1H3' in file]
    train_data = load_data(train_files, data_dir)
    test_data = load_data(test_files, data_dir)
    
    # Prepare training sequences
    input_size = 1
    learning_rate = 0.001
    epochs = 200
    batch_size =256
    length_testing=120
    # Prepare training data
    X_train, y_train = [], []
    for i in range(len(train_data) - sequence_length - output_size):
        X_train.append(train_data[i:i + sequence_length])
        y_train.append(train_data[i + sequence_length:i + sequence_length + output_size])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Normalize the input and output separately
    # scaler_input = MinMaxScaler(feature_range=(-1, 1))
    # X_train = scaler_input.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    
    # scaler_target = MinMaxScaler(feature_range=(-1, 1))
    # y_train = scaler_target.fit_transform(y_train)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Shape: (samples, sequence_length, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # Shape: (samples, output_size)
    
    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = CustomGRUModel(input_size, hidden_sizes, output_size).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)
    
    # Parameters for the weighted loss
    w = w  # Weight factor for the first x seconds
    x_time_steps = x_seconds * 20  # Since it's 20Hz, multiply by 20
    
    # Store loss values for analysis
    all_losses = []
    best_loss = float('inf')  # Track the best (minimum) loss
    print(f"Training started for : {model_name} \n")
    try:
        # Training Loop
        for epoch in range(epochs):
            total_loss = 0.0
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
                loss = loss_first_x_steps + loss_remaining_steps + 0.05 * continuity_loss+ 1e-8
                # Backward pass and optimization

                # Check for NaN in loss
                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at epoch {epoch}, batch {batch_idx}. Exiting training loop.")
                    return "Training stopped due to NaN loss."
            
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
                optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
                # Print progress every 100 batches
                if (batch_idx + 1) % 2000 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}/{len(train_loader)} "
                    f"Loss: {avg_loss:.8f}")
            # Calculate average loss for the epoch
            avg_epoch_loss = total_loss / len(train_loader)
            all_losses.append(avg_epoch_loss)
            
            # Save the model if the current loss is lower than the best loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                model_path = os.path.join(model_folder, model_name)
                torch.save(model.state_dict(), model_path)
            
            # Step the learning rate scheduler
            print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.8f} of model: {model_name}")
            scheduler.step()
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"Skipping model {model_name} due to insufficient GPU memory.")
            torch.cuda.empty_cache()  # Clear the GPU memory
            return f"Skipped model {model_name} due to memory constraints."
        else:
            raise  # Re-raise the exception if it's not a memory error

    # Plotting and saving the error vs. epoch plot
    plt.figure()
    plt.plot(all_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs. Epoch for {model_name}')
    plt.legend()
    plt.grid()

    # Save the plot with a unique name for each model
    folder_path = "brute_results"
    os.makedirs(folder_path, exist_ok=True)
    plot_filename = os.path.join(folder_path, f"{model_name}_error_vs_epoch.png")
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    print("Training complete")
    print(f"Testing started for : {model_name}")
    model_path = os.path.join(model_folder, model_name)
    if not os.path.exists(model_path):
        print(f"Model file {model_name} not found. Skipping testing phase.")
        return f"Training completed, but testing skipped for model: {model_name}"

    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare test data
    meters_to_cm = 25  # Conversion factor
    sequence_length = config['sequence_length']
    
    # Extract a portion of test data
    start_index = 20 * 1  # Starting index for data
    end_index = 20 * length_testing + sequence_length
    test_data_segment = test_data[start_index:end_index]
    
    # Variables for metrics and plotting
    prediction_times = []
    absolute_errors = []
    errors_3s = []
    errors_4s = []
    errors_5s = []
    steps_3s = 3 * 20  # 3 seconds at 20Hz
    steps_4s = 4 * 20  # 4 seconds at 20Hz
    steps_5s = 5 * 20  # 5 seconds at 20Hz
    predictions = []  # Store predictions for video
    abs_error_series = []  # Store error series for video

    with torch.no_grad():
        real_time_input = test_data_segment[:sequence_length]
        
        for i in range(sequence_length, len(test_data_segment) - output_size):
            # Prepare input tensor
            input_tensor = torch.tensor(real_time_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).cuda()
            # Model prediction
            start_time = time.time()
            predicted = model(input_tensor).cpu().numpy().flatten()
            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            
            # Calculate errors
            true_future = test_data_segment[i:i + output_size] * meters_to_cm
            predicted_future = predicted * meters_to_cm
            abs_error = np.abs(true_future - predicted_future)
            absolute_errors.append(abs_error.mean())
            errors_3s.append(np.mean(abs_error[:steps_3s]))
            errors_4s.append(np.mean(abs_error[:steps_4s]))
            errors_5s.append(np.mean(abs_error[:steps_5s]))
            
            # Store predictions and error for video
            predictions.append((true_future, predicted_future))
            abs_error_series.append(abs_error)
            
            # Shift the window
            real_time_input = np.append(real_time_input[1:], test_data_segment[i])
    
    # Create and save video of predictions and errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Setup the prediction plot
    line1, = ax1.plot([], [], label="True Future", color="blue")
    line2, = ax1.plot([], [], label="Predicted Future", color="orange")
    ax1.set_xlim(0, output_size)
    ax1.set_ylim(min(test_data_segment) * meters_to_cm, max(test_data_segment) * meters_to_cm)
    ax1.set_title(f"Testing Predictions for {model_name}")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Values (cm)")
    ax1.legend()
    ax1.grid()

    # Setup the error plot
    line3, = ax2.plot([], [], label="Absolute Error", color="red")
    ax2.set_xlim(0, output_size)
    ax2.set_ylim(0, max([np.max(e) for e in abs_error_series]) * 1.1)  # Add some padding
    ax2.set_title("Prediction Errors")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Error (cm)")
    ax2.legend()
    ax2.grid()

    # Update function for animation
    def update(frame):
        true, pred = predictions[frame]
        error = abs_error_series[frame]
        # Update the prediction plot
        line1.set_data(range(len(true)), true)
        line2.set_data(range(len(pred)), pred)
        # Update the error plot
        line3.set_data(range(len(error)), error)
        return line1, line2, line3

    # Create and save the animation
    ani = animation.FuncAnimation(fig, update, frames=len(predictions), blit=True)
    video_filename = os.path.join("brute_results", f"{model_name}_predictions_and_errors.mp4")
    writer = FFMpegWriter(fps=10, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(video_filename, writer=writer)
    plt.close(fig)

    # Calculate metrics
    avg_prediction_time = np.mean(prediction_times)
    avg_absolute_error = np.mean(absolute_errors)
    avg_error_3s = np.mean(errors_3s)
    avg_error_4s = np.mean(errors_4s)
    avg_error_5s = np.mean(errors_5s)
    
    # Save results to summary file
    summary_line = (
        f"Model: {model_name}\n"
        f"Hidden Sizes: {hidden_sizes}\n"
        f"Sequence Length: {sequence_length}\n"
        f"Output Size: {output_size}\n"
        f"x_seconds: {x_seconds}\n"
        f"w (weight): {w}\n"
        f"Average Prediction Time: {avg_prediction_time:.4f} seconds\n"
        f"Average Absolute Error for the first 3 seconds: {avg_error_3s:.4f} cm\n"
        f"Average Absolute Error for the first 4 seconds: {avg_error_4s:.4f} cm\n"
        f"Average Absolute Error for the first 5 seconds: {avg_error_5s:.4f} cm\n"
        f"Total Average Absolute Error: {avg_absolute_error:.4f} cm\n"
        f"Video File: {video_filename}\n"
        f"------------------------------------------------------------\n"
    )
    save_plot_with_summary(all_losses, summary_line, model_name)

    with open('model_summary.txt', 'a') as f:
        f.write(summary_line)
    
    return f"Completed training, testing, and video generation for model: {model_name}"



# # Function to assign GPU to each process
# def set_device(process_id, num_gpus):
#     # Assign GPU based on process_id
#     gpu_id = process_id % num_gpus
#     torch.cuda.set_device(gpu_id)
#     print(f"Process {process_id} assigned to GPU {gpu_id}")

# def run_in_batches(config_lines, num_gpus=2):
#     num_gpus = min(torch.cuda.device_count(), num_gpus)  # Get the number of available GPUs
#     if num_gpus == 0:
#         raise RuntimeError("No GPUs available for training")

#     # Limit the number of concurrent processes to the number of GPUs
#     max_concurrent_processes = num_gpus

#     # Create a list of arguments for each configuration
#     args_list = [(config_lines[i], i, num_gpus) for i in range(len(config_lines))]

#     # Use multiprocessing to run the processes, limiting to max_concurrent_processes
#     with mp.Pool(processes=max_concurrent_processes) as pool:
#         # Use the pool to map the function to the list of arguments
#         results = pool.starmap(train_and_test_model, args_list)

#     # Print completion messages
#     for res in results:
#         print(res)

if __name__ == '__main__':
    with open('model_configs4.txt', 'r') as f:
        config_lines = f.readlines()

    for config_line in config_lines:
        result = train_and_test_model(config_line)
        print(result)
