
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


# =========================================================
# 1. Configuration
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "multisubject_eeg.npz"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33]
SAMPLE_RATE = 160.0
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 32

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# 2. NeuroFusion-T architecture
# =========================================================

class NeuroFusionT(nn.Module):

    def __init__(self, n_channels, n_times, sample_rate=160.0):
        super().__init__()

        self.sample_rate = sample_rate

        # Spatial branch:
        # Learns channel relationships using full-channel filters.
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=(n_channels, 1),
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_projection = nn.Linear(16, 64)

        # Temporal branch:
        # Learns temporal patterns from all EEG channels.
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                n_channels, 32,
                kernel_size=15,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.3),

            nn.Conv1d(
                32, 64,
                kernel_size=15,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Frequency branch:
        # Differentiable FFT-based mu/alpha and beta features.
        self.frequency_projection = nn.Sequential(
            nn.Linear(n_channels * 2, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )

        # Attention-based fusion of three branch tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        self.fusion_norm = nn.LayerNorm(64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def frequency_features(self, x):
        # x shape: batch, channels, time

        n_times = x.shape[-1]

        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.abs().pow(2) / n_times

        frequencies = torch.fft.rfftfreq(
            n_times,
            d=1.0 / self.sample_rate
        ).to(x.device)

        mu_mask = (frequencies >= 8.0) & (frequencies < 13.0)
        beta_mask = (frequencies >= 13.0) & (frequencies <= 30.0)

        mu_power = power[:, :, mu_mask].mean(dim=-1)
        beta_power = power[:, :, beta_mask].mean(dim=-1)

        # Log power improves numerical stability.
        mu_power = torch.log(mu_power + 1e-10)
        beta_power = torch.log(beta_power + 1e-10)

        features = torch.cat(
            [mu_power, beta_power],
            dim=1
        )

        return features

    def forward(self, x):
        # Input shape: batch, 1, channels, time

        eeg = x.squeeze(1)

        # Spatial representation
        spatial = self.spatial_conv(x)
        spatial = self.spatial_pool(spatial).flatten(1)
        spatial = self.spatial_projection(spatial)

        # Temporal representation
        temporal = self.temporal_conv(eeg).squeeze(-1)

        # Frequency representation
        frequency = self.frequency_features(eeg)
        frequency = self.frequency_projection(frequency)

        # Three feature tokens
        tokens = torch.stack(
            [spatial, temporal, frequency],
            dim=1
        )

        attended, _ = self.attention(
            tokens, tokens, tokens,
            need_weights=False
        )

        fused = self.fusion_norm(
            tokens + attended
        )

        fused = fused.mean(dim=1)

        return self.classifier(fused)


# =========================================================
# 3. Data helpers
# =========================================================

def normalize_from_train(X_train, *other_sets):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)

    output = [(X_train - mean) / std]

    for item in other_sets:
        output.append((item - mean) / std)

    return output


def make_loader(X, y, shuffle):
    X = X[:, np.newaxis, :, :]

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


# =========================================================
# 4. Training helpers
# =========================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()


def evaluate_loss(model, loader, criterion):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            loss = criterion(model(xb), yb)
            total += loss.item() * xb.size(0)

    return total / len(loader.dataset)


def select_epochs(X_train, y_train, groups_train, seed):
    # Validation subjects are selected only from outer training subjects.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.22,
        random_state=2026
    )

    train_idx, val_idx = next(
        splitter.split(X_train, y_train, groups_train)
    )

    X_inner = X_train[train_idx]
    y_inner = y_train[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    X_inner, X_val = normalize_from_train(
        X_inner, X_val
    )

    train_loader = make_loader(
        X_inner, y_inner, True
    )

    val_loader = make_loader(
        X_val, y_val, False
    )

    set_seed(seed)

    model = NeuroFusionT(
        X_train.shape[1],
        X_train.shape[2],
        SAMPLE_RATE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        train_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss = evaluate_loss(
            model, val_loader, criterion
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0:
            print(
                f"  Epoch {epoch:02d} | "
                f"Validation loss: {val_loss:.4f}"
            )

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_weights)

    return best_epoch


def train_final_model(
    X_train,
    y_train,
    X_test,
    epochs,
    seed
):
    X_train_scaled, X_test_scaled = normalize_from_train(
        X_train, X_test
    )

    train_loader = make_loader(
        X_train_scaled, y_train, True
    )

    set_seed(seed)

    model = NeuroFusionT(
        X_train.shape[1],
        X_train.shape[2],
        SAMPLE_RATE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_epoch(
            model, train_loader, optimizer, criterion
        )

    model.eval()

    test_tensor = torch.tensor(
        X_test_scaled[:, np.newaxis, :, :],
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        predictions = model(test_tensor).argmax(1).cpu().numpy()

    return predictions, model


# =========================================================
# 5. Load dataset
# =========================================================

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"]

# Original event 2 = class 0; event 3 = class 1.
y = np.where(y == 2, 0, 1).astype(np.int64)

print("\nNEUROSYNC - NeuroFusion-T")
print("Device:", DEVICE)
print("Dataset:", X.shape)
print("Subjects:", np.unique(groups))
print("Seeds:", SEEDS)


# =========================================================
# 6. Subject-wise evaluation
# =========================================================

logo = LeaveOneGroupOut()

seed_predictions = {
    seed: np.zeros(len(y), dtype=np.int64)
    for seed in SEEDS
}

fold_records = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):

    subject = groups[test_idx][0]

    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    print("\n" + "=" * 55)
    print(f"Fold {fold}/10 | Unseen subject: {subject}")
    print("=" * 55)

    for seed in SEEDS:

        print(f"\nSeed: {seed}")

        epochs = select_epochs(
            X_train,
            y_train,
            groups_train,
            seed
        )

        predictions, model = train_final_model(
            X_train,
            y_train,
            X_test,
            epochs,
            seed
        )

        seed_predictions[seed][test_idx] = predictions

        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(y_test, predictions)

        print("Selected epochs:", epochs)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Balanced accuracy: {balanced * 100:.2f}%")

        fold_records.append(
            (subject, seed, epochs, accuracy, balanced)
        )


# =========================================================
# 7. Aggregate results
# =========================================================

print("\n" + "=" * 55)
print("NEUROFUSION-T FINAL RESULTS")
print("=" * 55)

scores = []

for seed in SEEDS:
    predictions = seed_predictions[seed]

    accuracy = accuracy_score(y, predictions)
    balanced = balanced_accuracy_score(y, predictions)

    scores.append((accuracy, balanced))

    print(
        f"Seed {seed} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"Balanced: {balanced * 100:.2f}%"
    )

accuracy_values = np.array([s[0] for s in scores])
balanced_values = np.array([s[1] for s in scores])

print("\nMean across seeds:")
print(
    f"Accuracy: {accuracy_values.mean() * 100:.2f}% "
    f"+/- {accuracy_values.std(ddof=1) * 100:.2f}%"
)
print(
    f"Balanced accuracy: {balanced_values.mean() * 100:.2f}% "
    f"+/- {balanced_values.std(ddof=1) * 100:.2f}%"
)

stacked = np.stack(
    [seed_predictions[s] for s in SEEDS]
)

ensemble = (
    np.sum(stacked, axis=0) >= 2
).astype(np.int64)

ensemble_accuracy = accuracy_score(y, ensemble)
ensemble_balanced = balanced_accuracy_score(y, ensemble)

print("\nMajority vote:")
print(f"Accuracy: {ensemble_accuracy * 100:.2f}%")
print(f"Balanced accuracy: {ensemble_balanced * 100:.2f}%")

for seed in SEEDS:
    print(f"\nConfusion matrix - seed {seed}:")
    print(confusion_matrix(y, seed_predictions[seed]))


# =========================================================
# 8. Save predictions and report
# =========================================================

np.savez(
    OUTPUT_DIR / "neurofusion_t_predictions.npz",
    y_true=y,
    **{
        f"seed_{seed}": seed_predictions[seed]
        for seed in SEEDS
    },
    majority_vote=ensemble
)

report_path = OUTPUT_DIR / "neurofusion_t_results.txt"

with open(report_path, "w") as file:

    file.write("NEUROSYNC - NeuroFusion-T Results\n\n")
    file.write(f"Dataset shape: {X.shape}\n")
    file.write(f"Subjects: {len(np.unique(groups))}\n")
    file.write(f"Seeds: {SEEDS}\n\n")

    for seed, (acc, bal) in zip(SEEDS, scores):
        file.write(
            f"Seed {seed}: accuracy={acc * 100:.2f}%, "
            f"balanced_accuracy={bal * 100:.2f}%\n"
        )

    file.write(
        f"\nMean accuracy: {accuracy_values.mean() * 100:.2f}%\n"
    )
    file.write(
        f"Accuracy SD: {accuracy_values.std(ddof=1) * 100:.2f}%\n"
    )
    file.write(
        f"Mean balanced accuracy: "
        f"{balanced_values.mean() * 100:.2f}%\n"
    )
    file.write(
        f"Balanced accuracy SD: "
        f"{balanced_values.std(ddof=1) * 100:.2f}%\n"
    )
    file.write(
        f"\nMajority vote accuracy: "
        f"{ensemble_accuracy * 100:.2f}%\n"
    )
    file.write(
        f"Majority vote balanced accuracy: "
        f"{ensemble_balanced * 100:.2f}%\n"
    )

print("\nResults saved:", report_path)
print("Predictions saved: outputs/neurofusion_t_predictions.npz")
print("\nNeuroFusion-T experiment completed.")
