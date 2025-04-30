import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import re
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation

# Ensure directories exist
model_dir = 'GRU_models_seq'
os.makedirs(model_dir, exist_ok=True)
plot_dir = 'prediction_videos'
os.makedirs(plot_dir, exist_ok=True)

# Specify the GPU index (0 or 1)
gpu_index = 0 # Set to 1 if you want to use the second GPU

# Set the environment variable to specify the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

# Read model configurations from file
config_file_path = 'model_configs_seq2.txt'
with open(config_file_path, 'r') as f:
    model_configs = [line.strip() for line in f if line.strip()]

# Parse configuration string
def parse_config_line(config_line):
    config = {}
    params = config_line.split(';')
    for param in params:
        key, value = param.split(':')
        key = key.strip()
        value = value.strip()
        if key == 'hidden_sizes':
            config[key] = list(map(int, re.findall(r'\d+', value)))
        elif key in ['sequence_length', 'output_size', 'x_seconds', 'w', 'batch_size', 'epochs']:
            config[key] = int(value)
        elif key in ['learning_rate', 'dropout_p']:
            config[key] = float(value)
    return config

# Define GRU model class with dropout
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160, dropout_p=0.5):
        super(GRUModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.gru_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru_layers.append(nn.GRU(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_sizes[i-1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x, h):
        h_out = []
        out = x
        for i, gru in enumerate(self.gru_layers):
            out, h_i = gru(out, h[i])
            h_out.append(h_i)
            if i < self.num_layers - 1:
                out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.tanh(out)
        return out, h_out
        
    def init_hidden(self, batch_size):
        h = [torch.zeros(1, batch_size, hidden_size).to(device) for hidden_size in self.hidden_sizes]
        return h

# Training and testing loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess training data
def load_data(file_list, data_dir='train_data_normalised'):
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.iloc[:, 0].values)
    return np.array(data)

# Load test data
test_file_path = 'train_data_normalised/D1H3_normalised.csv'
test_data = pd.read_csv(test_file_path, header=None).iloc[:, 0].values
meters_to_cm = 25  # Conversion factor

# Iterate through model configurations
with open('training_summary.txt', 'a') as summary_file:
    for config_line in model_configs:
        config = parse_config_line(config_line)
        print(f"Training model with config: {config}")
        
        # Set configuration variables
        sequence_length = config['sequence_length']
        output_size = config['output_size']
        hidden_sizes = config['hidden_sizes']
        x_seconds = config['x_seconds']
        w = config['w']
        batch_size = config['batch_size']
        epochs = config['epochs']
        learning_rate = config['learning_rate']
        dropout_p = config.get('dropout_p', 0.5)  # Default dropout probability
        model_name = f"D1_new_GRU_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}.pth"

        # Initialize model, criterion, optimizer
        model = GRUModel(input_size=1, hidden_sizes=hidden_sizes, output_size=output_size, dropout_p=dropout_p).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)

        # Prepare training data (assuming training data has been loaded)
        csv_files = os.listdir('train_data_normalised')
        train_files = [file for file in csv_files if 'D1H' in file and 'D1H3' not in file]
        train_data = load_data(train_files)

        X_train, y_train = [], []
        for i in range(len(train_data) - sequence_length - output_size):
            X_train.append(train_data[i:i + sequence_length])
            y_train.append(train_data[i + sequence_length:i + sequence_length + output_size])

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print("Starting training for model: ",hidden_sizes)
                # Training loop
        start_time = time.time()
        all_losses = []
        best_loss = float('inf')
        x_time_steps = x_seconds * 20  # Steps for weighted loss
        stop_training_due_to_nan = False  # Flag to stop training for this config

        for epoch in range(epochs):
            total_loss = 0.0
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                h = model.init_hidden(inputs.size(0))
                optimizer.zero_grad()
                outputs, h = model(inputs, h)
                h = [h_i.detach() for h_i in h]
                
                first_x_steps = targets[:, :x_time_steps]
                remaining_steps = targets[:, x_time_steps:]
                loss_first_x_steps = criterion(outputs[:, :x_time_steps], first_x_steps) * w
                loss_remaining_steps = criterion(outputs[:, x_time_steps:], remaining_steps) if remaining_steps.size(1) > 0 else 0
                continuity_loss = torch.mean((outputs[:, 1:] - outputs[:, :-1]) ** 2)
                loss = loss_first_x_steps + loss_remaining_steps + 0.1 * continuity_loss

                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at epoch {epoch + 1}. Skipping model configuration.")
                    stop_training_due_to_nan = True
                    break  # Exit the inner loop

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # If NaN detected, stop further training for this configuration
            if stop_training_due_to_nan:
                break

            avg_epoch_loss = total_loss / len(train_loader)
            all_losses.append(avg_epoch_loss)

            # Print the error for the current epoch
            if epoch%20==0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.6f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                model_name = f"D1_new_GRU_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}.pth"
                model_path = os.path.join(model_dir, model_name)
                print("New model saved with loss: ", best_loss, " at: ", model_path)                
                torch.save(model.state_dict(), model_path)

            scheduler.step()

        # Stop training for this configuration and move to the next one if NaN was detected
        if stop_training_due_to_nan:
            summary_file.write(
                f"Model: {model_name}\n"
                f"Training stopped due to NaN loss.\n"
                f"------------------------------------------------------------\n"
            )
            print(f"Training stopped for model: {model_name} due to NaN loss.")
            continue  # Move to the next configuration


        # for epoch in range(epochs):
        #     total_loss = 0.0
        #     model.train()
        #     for inputs, targets in train_loader:
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         h = model.init_hidden(inputs.size(0))
        #         optimizer.zero_grad()
        #         outputs, h = model(inputs, h)
        #         h = [h_i.detach() for h_i in h]
                
        #         first_x_steps = targets[:, :x_time_steps]
        #         remaining_steps = targets[:, x_time_steps:]
        #         loss_first_x_steps = criterion(outputs[:, :x_time_steps], first_x_steps) * w
        #         loss_remaining_steps = criterion(outputs[:, x_time_steps:], remaining_steps) if remaining_steps.size(1) > 0 else 0
        #         continuity_loss = torch.mean((outputs[:, 1:] - outputs[:, :-1]) ** 2)
        #         loss = loss_first_x_steps + loss_remaining_steps + 0.1 * continuity_loss
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()
                        
        #     avg_epoch_loss = total_loss / len(train_loader)
        #     all_losses.append(avg_epoch_loss)
        #     if avg_epoch_loss < best_loss:
        #         best_loss = avg_epoch_loss
        #         model_name = f"D1_GRU_{sequence_length//20}_{output_size//20}_{'_'.join(map(str, hidden_sizes))}.pth"
        #         model_path = os.path.join(model_dir, model_name)
        #         torch.save(model.state_dict(), model_path)
        #     scheduler.step()

        training_time = time.time() - start_time
        print(f"Model trained in {training_time:.2f} seconds. Best loss: {best_loss:.4f}")
        print("starting TESTING")
        # Test the model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        prediction_times, absolute_errors = [], []
        errors_3s, errors_4s, errors_5s = [], [], []
        steps_3s, steps_4s, steps_5s = 3 * 20, 4 * 20, 5 * 20  # Steps for 3s, 4s, 5s

        total_steps = len(test_data) - sequence_length - output_size - 1
        start_index = sequence_length
        end_index = start_index + total_steps
        if end_index + output_size > len(test_data):
            end_index = len(test_data) - output_size
            total_steps = end_index - start_index

        fig, ax = plt.subplots(figsize=(12, 8))
        metadata = dict(title='Prediction Video', artist='Matplotlib', comment='Prediction over time')
        writer = animation.FFMpegWriter(fps=20, metadata=metadata)
        model_name_video = model_name.replace('.pth', '.mp4')
        video_path = os.path.join(plot_dir, model_name_video)
        with writer.saving(fig, video_path, dpi=100):
            h = model.init_hidden(1)
            for i in range(start_index, end_index):
                input_seq = test_data[i - sequence_length:i]
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                start_pred_time = time.time()
                with torch.no_grad():
                    output, h = model(input_tensor, h)
                    h = [h_i.detach() for h_i in h]
                prediction_times.append(time.time() - start_pred_time)
                predicted = output.cpu().numpy().flatten()
                true_future = test_data[i:i + output_size] * meters_to_cm
                predicted_future = predicted * meters_to_cm
                abs_error = np.abs(true_future - predicted_future)

                absolute_errors.append(abs_error.mean())
                errors_3s.append(np.mean(abs_error[:steps_3s]))
                errors_4s.append(np.mean(abs_error[:steps_4s]))
                errors_5s.append(np.mean(abs_error[:steps_5s]))

                # Plot prediction and error
                fig.clear()

                # Plot prediction
                ax1 = fig.add_subplot(2, 1, 1)
                time_axis = np.arange(i, i + output_size) / 20
                ax1.plot(time_axis, true_future, 'g--', label='True Future Data (cm)')
                ax1.plot(time_axis, predicted_future, 'r', label='Predicted Data (cm)')
                ax1.set_ylim(-30, 30)
                ax1.set_title(f"Time Elapsed: {(i - start_index) / 20:.2f} s")
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Prediction (cm)')
                ax1.legend()

                # Plot error
                ax2 = fig.add_subplot(2, 1, 2)
                ax2.plot(time_axis, abs_error, 'b', label='Absolute Error (cm)')
                ax2.set_ylim(0, 15)
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Error (cm)')
                ax2.legend()

                # Grab the frame for video
                writer.grab_frame()

        avg_prediction_time = np.mean(prediction_times) if prediction_times else 0
        avg_error = np.mean(absolute_errors) if absolute_errors else 0
        avg_error_3s = np.mean(errors_3s) if errors_3s else 0
        avg_error_4s = np.mean(errors_4s) if errors_4s else 0
        avg_error_5s = np.mean(errors_5s) if errors_5s else 0

        summary_file.write(
            f"Model: {model_name}\n"
            f"Average Prediction Time: {avg_prediction_time:.4f} s\n"
            f"Average Absolute Error (Total): {avg_error:.4f} cm\n"
            f"Average Absolute Error (3s): {avg_error_3s:.4f} cm\n"
            f"Average Absolute Error (4s): {avg_error_4s:.4f} cm\n"
            f"Average Absolute Error (5s): {avg_error_5s:.4f} cm\n\n"
            f"------------------------------------------------------------\n"
        )
        print(f"Model: {model_name} tested and results saved.\n")
