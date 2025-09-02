# torchscript_converter.py
import torch
from typing import List
from collections import OrderedDict

# If you want to sanity-check key shapes against your original class, you can import it:
# from LSTM_brute import LSTMModel  # not required for scripting; only for reference

# ====== CONFIG (edit these) ====================================================
CKPT_PT      = "noisyLSTM_models_seq/noisy_D1_LSTM_40_6_1024_512/epoch_400.pt"  # full checkpoint .pt (dict) or set to None
WEIGHTS_PTH  = None  # e.g., "noisyLSTM_models_seq/.../epoch_400.pth" if you only have weights
HIDDEN_SIZES = [1024, 512]  # must match training
OUTPUT_SIZE  = 120          # steps (e.g., 6s @ 20 Hz)
DEVICE       = torch.device("cpu")  # export on CPU for portability
OUT_FILE     = "epoch_400_script.pt"
# ==============================================================================

class LSTMCoreScriptable(torch.nn.Module):
    """
    Script-friendly LSTM twin:
      - Keeps param names/shapes compatible with your trained model
      - Uses FLAT List[Tensor] state: [h0, c0, h1, c1, ...]
    """
    def __init__(self, input_size: int = 1, hidden_sizes: List[int] = None, output_size: int = 160):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        lstms: List[torch.nn.Module] = []
        lstms.append(torch.nn.LSTM(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        for i in range(1, self.num_layers):
            lstms.append(torch.nn.LSTM(hidden_sizes[i - 1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.lstm_layers = torch.nn.ModuleList(lstms)

        self.fc   = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, flat_state: List[torch.Tensor]):
        # x: [B, T, 1]; flat_state: [h0,c0,h1,c1,...]
        out = x
        next_flat: List[torch.Tensor] = []
        idx = 0
        # IMPORTANT: iterate over modules directly (TorchScript does not allow dynamic indexing)
        for lstm in self.lstm_layers:
            h_i = flat_state[idx];   idx += 1
            c_i = flat_state[idx];   idx += 1
            out, hc = lstm(out, (h_i, c_i))
            # unpack in script-friendly way
            h_o = hc[0]
            c_o = hc[1]
            next_flat.append(h_o)
            next_flat.append(c_o)
        out = out[:, -1, :]
        out = self.tanh(self.fc(out))
        return out, next_flat

def load_trained_state():
    # Prefer weights-only file if provided; else load full checkpoint
    if WEIGHTS_PTH is not None:
        # weights-only .pth: a plain state_dict
        return torch.load(WEIGHTS_PTH, map_location=DEVICE)
    elif CKPT_PT is not None:
        # full checkpoint .pt (dict). Use weights_only=True to silence the warning on 2.4+
        try:
            ckpt = torch.load(CKPT_PT, map_location=DEVICE, weights_only=True)  # PyTorch >=2.4
        except TypeError:
            ckpt = torch.load(CKPT_PT, map_location=DEVICE)  # fallback for older versions
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        # sometimes people accidentally save raw state_dict in .pt
        return ckpt
    else:
        raise ValueError("Provide either CKPT_PT or WEIGHTS_PTH")

def main():
    # Build scriptable twin and load weights
    twin = LSTMCoreScriptable(input_size=1, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE).to(DEVICE)
    sd = load_trained_state()

    # Filter to known keys (robust against optimizer/extra entries)
    sd_clean = OrderedDict((k, v) for k, v in sd.items() if k in twin.state_dict())
    missing, unexpected = twin.load_state_dict(sd_clean, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    twin.eval()

    # Example inputs for scripting validation
    B = 1
    x_ex = torch.zeros(1, 1, 1, dtype=torch.float32, device=DEVICE)
    flat_state_ex: List[torch.Tensor] = []
    for hs in HIDDEN_SIZES:
        flat_state_ex.append(torch.zeros(1, B, hs, device=DEVICE))  # h
        flat_state_ex.append(torch.zeros(1, B, hs, device=DEVICE))  # c
    for lstm in twin.lstm_layers:
        lstm.flatten_parameters()

    # Script and quick dry-run
    scripted = torch.jit.script(twin)
    _y, _st = scripted(x_ex, flat_state_ex)  # type: ignore[func-returns-value]
    scripted.save(OUT_FILE)
    print(f"Saved TorchScript to: {OUT_FILE}")

if __name__ == "__main__":
    main()
