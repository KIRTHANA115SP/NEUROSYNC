
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------
# Configuration
# --------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "multisubject_eeg.npz"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "neurosync_eegnet_ensemble.pt"

SEEDS = [11, 22, 33]
EPOCHS = 8
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

torch.set_num_threads(4)


# --------------------------------
# Reproducibility
# --------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------
# EEGNet architecture
# --------------------------------

class EEGNet(nn.Module):

    def __init__(self, n_channels, n_times, n_classes=2):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1, 8, (1, 64),
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(8)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                8, 16,
                (n_channels, 1),
                groups=8,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                16, 16,
                (1, 16),
                padding=(0, 8),
                groups=16,
                bias=False
            ),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1, 1, n_channels, n_times
            )
            size = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(size, n_classes)

    def features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.separable(x)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


# --------------------------------
# Load dataset
# --------------------------------

print("Loading NEUROSYNC dataset...")

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

# Original labels: 2 -> left, 3 -> right
y = np.where(y == 2, 0, 1).astype(np.int64)

n_samples, n_channels, n_times = X.shape

print("Dataset shape:", X.shape)
print("Total samples:", n_samples)
print("Channels:", n_channels)
print("Time samples:", n_times)


# --------------------------------
# Final normalization
# --------------------------------

# Fit normalization using all data.
# This is appropriate for the final demonstration model,
# after the independent LOSO evaluation is complete.

channel_mean = X.mean(
    axis=(0, 2),
    keepdims=True
)

channel_std = X.std(
    axis=(0, 2),
    keepdims=True
)

channel_std = np.maximum(channel_std, 1e-8)

X_normalized = (X - channel_mean) / channel_std

X_tensor = torch.tensor(
    X_normalized[:, np.newaxis, :, :],
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.long
)

dataset = TensorDataset(X_tensor, y_tensor)


# --------------------------------
# Train three EEGNet models
# --------------------------------

model_states = {}

for seed in SEEDS:

    print("\n" + "=" * 50)
    print(f"Training EEGNet | Seed {seed}")
    print("=" * 50)

    set_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = EEGNet(
        n_channels=n_channels,
        n_times=n_times
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            loss = criterion(model(xb), yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        average_loss = total_loss / len(dataset)

        print(
            f"Seed {seed} | "
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {average_loss:.4f}"
        )

    model.eval()

    model_states[seed] = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    print(f"Seed {seed} training completed.")


# --------------------------------
# Save final model package
# --------------------------------

channels = []

if "channels" in data.files:
    for channel in data["channels"].tolist():
        if isinstance(channel, bytes):
            channels.append(channel.decode("utf-8"))
        else:
            channels.append(str(channel))

sampling_rate = None

if "sampling_rate" in data.files:
    sampling_rate = float(
        np.asarray(data["sampling_rate"]).item()
    )

checkpoint = {
    "model_name": "NEUROSYNC EEGNet Ensemble",
    "architecture": "EEGNet",
    "model_states": model_states,
    "seeds": SEEDS,
    "epochs": EPOCHS,
    "n_channels": n_channels,
    "n_times": n_times,
    "n_classes": 2,
    "class_labels": {
        0: "Left-hand motor imagery",
        1: "Right-hand motor imagery"
    },
    "channel_mean": torch.tensor(channel_mean),
    "channel_std": torch.tensor(channel_std),
    "channels": channels,
    "sampling_rate": sampling_rate
}

torch.save(checkpoint, MODEL_PATH)

print("\n" + "=" * 50)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 50)

print("Models trained:", SEEDS)
print("Training epochs:", EPOCHS)
print("Saved model:", MODEL_PATH)
print("Normalization statistics saved.")
print("Model metadata saved.")
