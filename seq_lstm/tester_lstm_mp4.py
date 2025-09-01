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

# ------------------------------ INPUTS ------------------------------------
test_time  = 150  # seconds to simulate
model_path = 'seq_lstm/noisyLSTM_models_seq/noisy_D1_LSTM_40_6_1024_512/epoch_400.pth'
test_file_path = 'seq/train_data_normalised/D1H3_normalised.csv'  # input CSV

noise_std     = 0.05  # Gaussian noise sigma to add to input at test-time
sampling_rate = 20    # Hz (video fps & pacing)

# ----------------------------- HELPERS ------------------------------------
NAME_RATE_HZ = 20  # how many samples-per-second the <seq> and <out> fields in the model name are encoded at

def parse_model_path(model_path):
    """
    Parse sequence length, output size and hidden_sizes from a path that contains
    '_(GRU|LSTM)_<seq>_<out>_<hs...>' where <seq>,<out> are in SECONDS at NAME_RATE_HZ
    and <hs...> is '_' joined hidden sizes.
    """
    m = re.search(r'_(GRU|LSTM)_(\d+)_(\d+)_([\d_]+)', model_path)
    if not m:
        raise ValueError("Model path does not match expected pattern '_(GRU|LSTM)_<seq>_<out>_<hs...>'.")
    sequence_length = int(m.group(2)) * NAME_RATE_HZ  # seconds → steps @NAME_RATE_HZ
    output_size     = int(m.group(3)) * NAME_RATE_HZ  # seconds → steps @NAME_RATE_HZ
    hidden_sizes    = list(map(int, m.group(4).split('_')))
    return sequence_length, output_size, hidden_sizes

def build_output_path(model_path, sequence_length, output_size, hidden_sizes, noise_std, test_time):
    """Compose an informative MP4 path under seq/noisyprediction_videos/."""
    out_dir = os.path.join("seq", "noisyprediction_videos")
    os.makedirs(out_dir, exist_ok=True)
    parent_dir = os.path.basename(os.path.dirname(model_path))  # e.g., noisy_D1_LSTM_40_6_2048_1024
    pth_base   = os.path.splitext(os.path.basename(model_path))[0]  # e.g., epoch_100
    fname = f"{parent_dir}_{pth_base}_{sampling_rate}_Hz.mp4"
    return os.path.join(out_dir, fname)

# ------------------------------ MODEL -------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[512, 256], output_size=160):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers   = len(hidden_sizes)
        self.lstm_layers  = nn.ModuleList()
        # first layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        # stacked layers
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.fc   = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, state):
        """
        x:     [B, T, 1] (we'll use T=1 for streaming)
        state: list of (h_i, c_i) for each layer; each is [1, B, H_i]
        """
        next_state = []
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            h_i, c_i = state[i]
            out, (h_o, c_o) = lstm(out, (h_i, c_i))
            next_state.append((h_o, c_o))
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        out = self.tanh(out)
        return out, next_state

    def init_hidden(self, batch_size):
        return [
            (torch.zeros(1, batch_size, hs, device=device),
             torch.zeros(1, batch_size, hs, device=device))
            for hs in self.hidden_sizes
        ]

    @staticmethod
    def detach_state(state):
        return [(h.detach(), c.detach()) for (h, c) in state]

# ---------------------------- SETUP ---------------------------------------
sequence_length, output_size, hidden_sizes = parse_model_path(model_path)
os.makedirs("predictions", exist_ok=True)  # harmless (compat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMModel(input_size=1, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load test data (first column)
test_data = pd.read_csv(test_file_path, header=None).iloc[:, 0].values.astype(np.float32)

meters_to_cm = 25.0  # conversion factor used in your pipeline

# ---------------------------- RANGES --------------------------------------
total_steps = int(test_time * sampling_rate)
start_index = max(sequence_length, output_size)
end_index   = start_index + total_steps
if end_index + output_size > len(test_data):
    end_index   = len(test_data) - output_size
    total_steps = end_index - start_index

print(f"Testing for {total_steps / sampling_rate:.2f} seconds.")
print(f"start_index: {start_index} | end_index: {end_index} | len(test_data): {len(test_data)}")

desired_interval = 1.0 / sampling_rate

# Errors & timing
prediction_times = []
absolute_errors  = []
errors_3s, errors_4s, errors_5s = [], [], []
steps_3s, steps_4s, steps_5s = 3*sampling_rate, 4*sampling_rate, 5*sampling_rate

# Hidden states (list of (h,c))
h = model.init_hidden(batch_size=1)

# -------------------------- VIDEO WRITER ----------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Relative-time axes (fixed window): history (-Tin..0), future (0..Tout)
Tin_s  = sequence_length / sampling_rate
Tout_s = output_size     / sampling_rate
t_hist   = np.arange(-sequence_length, 0, 1, dtype=float) / sampling_rate            # [-Tin, ..., -1/sr]
t_future = np.arange(1, output_size + 1, 1, dtype=float) / sampling_rate             # [1/sr, ..., Tout]

out_mp4 = build_output_path(model_path, sequence_length, output_size, hidden_sizes, noise_std, test_time)
metadata = dict(title='LSTM forecast', artist='Matplotlib', comment='noisy test-time predictions')
writer   = FFMpegWriter(fps=sampling_rate, metadata=metadata)

# --------------------------- MAIN LOOP ------------------------------------
testing_start_time = time.time()
error_start_index  = start_index + sequence_length  # (kept from your GRU tester)
noisy_history = []  # store exactly what we fed (noisy) for plotting the history

with writer.saving(fig, out_mp4, dpi=300):
    with torch.no_grad():
        for i in range(start_index, end_index):
            iter_start = time.time()

            # ----- noisy input (test-time) -----
            clean_val = torch.tensor([[[test_data[i]]]], dtype=torch.float32, device=device)  # [1,1,1]
            noisy_val = clean_val + torch.randn_like(clean_val) * noise_std
            noisy_history.append(noisy_val.item())   # record for plotting
            if len(noisy_history) > sequence_length: # keep only last Tin samples
                noisy_history.pop(0)

            # forward pass (stateful, one sample per tick)
            t0 = time.perf_counter()
            output, h = model(noisy_val, h)
            h = model.detach_state(h)
            t1 = time.perf_counter()

            pred_time = t1 - t0
            prediction_times.append(pred_time)

            predicted = output.detach().cpu().numpy().flatten()
            true_future_cm      = test_data[i+1:i+1+output_size] * meters_to_cm
            predicted_future_cm = predicted * meters_to_cm
            abs_error           = np.abs(true_future_cm - predicted_future_cm)

            if i >= error_start_index:
                absolute_errors.append(abs_error.mean())
                errors_3s.append(np.mean(abs_error[:steps_3s]))
                errors_4s.append(np.mean(abs_error[:steps_4s]))
                errors_5s.append(np.mean(abs_error[:steps_5s]))

            # ------------- PLOT (time-based, sliding window) -------------
            ax1.clear(); ax2.clear()

            # plot noisy history (what the model actually saw)
            hist_cm = np.array(noisy_history) * meters_to_cm
            t_hist  = np.arange(-len(hist_cm), 0) / sampling_rate
            ax1.plot(t_hist, hist_cm, label='Noisy Input (cm)')

            # future (clean true vs pred)
            t_future = np.arange(1, output_size+1) / sampling_rate
            ax1.plot(t_future, true_future_cm, 'g--', label='True future (cm)')
            ax1.plot(t_future, predicted_future_cm, 'r', label='Predicted (cm)')
            ax1.axvline(0.0, linestyle=':', linewidth=1)

            Tin_s  = sequence_length / sampling_rate
            Tout_s = output_size     / sampling_rate
            ax1.set_xlim(-Tin_s, Tout_s)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (cm)')
            ax1.legend(loc='upper left')

            # overlay elapsed + timing
            total_elapsed = time.time() - testing_start_time
            avg_ms = (np.mean(prediction_times)*1000.0) if prediction_times else 0.0
            cur_ms = pred_time*1000.0
            fig.suptitle(
                f"Elapsed: {total_elapsed:.2f}s / {test_time}s   |   "
                f"Model: {os.path.basename(os.path.dirname(model_path))}/{os.path.basename(model_path)}",
                fontsize=12
            )
            ax1.text(
                0.99, 0.02,
                f"Pred time: {cur_ms:.2f} ms  (avg {avg_ms:.2f} ms)\nNoise σ={noise_std:.3f}",
                transform=ax1.transAxes, ha='right', va='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3')
            )

            # error subplot
            ax2.plot(t_future, abs_error, 'b', label='Absolute error (cm)')
            ax2.axvline(0.0, linestyle=':', linewidth=1)
            ax2.set_xlim(0.0, Tout_s)
            ax2.set_ylim(0.0, 18)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Error (cm)')
            ax2.legend(loc='upper left')

            writer.grab_frame()

            # pacing (optional; keeps real-time feel in the video)
            elapsed = time.time() - iter_start
            sleep_t = (1.0/sampling_rate) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

plt.close(fig)

# --------------------------- METRICS & PRINTS -----------------------------
if absolute_errors:
    avg_prediction_time = float(np.mean(prediction_times))
    avg_absolute_error  = float(np.mean(absolute_errors))
    avg_error_3s        = float(np.mean(errors_3s))
    avg_error_4s        = float(np.mean(errors_4s))
    avg_error_5s        = float(np.mean(errors_5s))

    print(f"\nSaved video: {out_mp4}")
    print(f"Average Prediction Time: {avg_prediction_time:.4f} s")
    print(f"Average Absolute Error (first 3s): {avg_error_3s:.4f} cm")
    print(f"Average Absolute Error (first 4s): {avg_error_4s:.4f} cm")
    print(f"Average Absolute Error (first 5s): {avg_error_5s:.4f} cm")
    print(f"Total Average Absolute Error: {avg_absolute_error:.4f} cm")
else:
    print("\nNo errors were recorded. Ensure test duration is sufficient and indices are in range.")
    print(f"Saved video: {out_mp4}")
