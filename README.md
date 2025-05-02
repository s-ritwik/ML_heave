# Ship Heave Prediction using GRU

This repository focuses on training GRU (Gated Recurrent Unit) models for predicting the **heave motion** data of a ship deck. It also includes comparative studies using TCNs and organizes the workflow into modular components for data processing, model training, and results evaluation.

---

## 📁 Folder Structure

### [GRU_generalised](./GRU_generalised)
Contains generalized GRU model implementations with different hyperparameter settings and input variations.

➡️ _[It considers input data to be an array and does prediction over them, this increases computation but increases the accuracy and does require hovering for a long time]_

---

### [TCN](./TCN)
Includes Temporal Convolutional Networks (TCNs) to serve as a benchmark model for sequence prediction.

➡️ _[It was intially considered, but too much computation for my jetson]_

---

### [results](./results)
Stores generated outputs from models including plots, metrics, and saved weights.

➡️ _[]_

---

### [seq](./seq)
Contains input datasets for heave motion in sequential format suitable for time-series modeling.

➡️ _[It considers input data to be an single data point and does prediction over them, this makes it computationally very effective but sudden data jumps due to bad estimations decreases accuracy]_

---

## 🛠️ Installation

Clone the repo and install tf and torch

```bash
git clone https://github.com/s-ritwik/ML_heave.git
