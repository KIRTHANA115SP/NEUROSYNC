
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


# -----------------------------
# 1. Reproducibility and setup
# -----------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(4)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "multisubject_eeg.npz"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("NEUROSYNC - EEGNet LOSO Experiment")
print("Device:", DEVICE)


# -----------------------------
# 2. Load dataset
# -----------------------------

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"]

# Convert original event labels 2 and 3 into 0 and 1
y = np.where(y == 2, 0, 1).astype(np.int64)

print("\nDataset shape:", X.shape)
print("Subjects:", np.unique(groups))
print("Classes:", np.unique(y, return_counts=True))

n_channels = X.shape[1]
n_times = X.shape[2]


# -----------------------------
# 3. EEGNet architecture
# -----------------------------

class EEGNet(nn.Module):

    def __init__(
        self,
        n_channels,
        n_times,
        n_classes=2,
        F1=8,
        D=2,
        F2=16,
        dropout=0.5
    ):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(F1)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                F1 * D,
                F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False
            ),
            nn.Conv2d(
                F1 * D,
                F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Calculate flattened feature size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            features = self._extract_features(dummy)
            flattened_size = features.flatten(1).shape[1]

        self.classifier = nn.Linear(
            flattened_size,
            n_classes
        )

    def _extract_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x):
        x = self._extract_features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# -----------------------------
# 4. Training function
# -----------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    max_epochs=40,
    patience=7
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(max_epochs):

        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                outputs = model(xb)
                loss = criterion(outputs, yb)

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0

        else:

            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)

    return model


# -----------------------------
# 5. Leave-One-Subject-Out
# -----------------------------

all_predictions = []
all_targets = []
subject_results = []

unique_subjects = np.unique(groups)

for test_subject in unique_subjects:

    print("\n" + "=" * 50)
    print("Testing unseen subject:", test_subject)
    print("=" * 50)

    test_mask = groups == test_subject

    X_test = X[test_mask]
    y_test = y[test_mask]

    X_train_all = X[~test_mask]
    y_train_all = y[~test_mask]
    groups_train_all = groups[~test_mask]

    # Split training subjects into training and validation subjects
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.22,
        random_state=SEED
    )

    train_idx, val_idx = next(
        splitter.split(
            X_train_all,
            y_train_all,
            groups_train_all
        )
    )

    X_train = X_train_all[train_idx]
    y_train = y_train_all[train_idx]

    X_val = X_train_all[val_idx]
    y_val = y_train_all[val_idx]

    # Fit channel normalization using training data only
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)

    channel_std = np.maximum(channel_std, 1e-8)

    X_train = (X_train - channel_mean) / channel_std
    X_val = (X_val - channel_mean) / channel_std
    X_test_scaled = (X_test - channel_mean) / channel_std

    # EEGNet expects (samples, 1, channels, time)
    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    X_test_scaled = X_test_scaled[:, np.newaxis, :, :]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    model = EEGNet(
        n_channels=n_channels,
        n_times=n_times
    ).to(DEVICE)

    model = train_model(
        model,
        train_loader,
        val_loader
    )

    # Evaluate on the unseen subject
    model.eval()

    test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(test_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    print(f"Subject accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_acc * 100:.2f}%")

    subject_results.append(
        (test_subject, accuracy, balanced_acc)
    )

    all_predictions.extend(predictions.tolist())
    all_targets.extend(y_test.tolist())


# -----------------------------
# 6. Final evaluation
# -----------------------------

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

overall_accuracy = accuracy_score(
    all_targets,
    all_predictions
)

overall_balanced = balanced_accuracy_score(
    all_targets,
    all_predictions
)

cm = confusion_matrix(
    all_targets,
    all_predictions
)

print("\n" + "=" * 50)
print("FINAL EEGNet LOSO RESULTS")
print("=" * 50)

print(f"Accuracy: {overall_accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {overall_balanced * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nSubject-wise results:")

for subject, acc, bal in subject_results:
    print(
        f"Subject {subject}: "
        f"Accuracy = {acc * 100:.2f}% | "
        f"Balanced Accuracy = {bal * 100:.2f}%"
    )


# -----------------------------
# 7. Save results
# -----------------------------

results_path = OUTPUT_DIR / "eegnet_loso_results.txt"

with open(results_path, "w") as file:

    file.write("NEUROSYNC - EEGNet LOSO Results\n\n")

    file.write(
        f"Accuracy: {overall_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Balanced Accuracy: {overall_balanced * 100:.2f}%\n\n"
    )

    file.write("Confusion Matrix:\n")
    file.write(str(cm) + "\n\n")

    file.write("Subject-wise Results:\n")

    for subject, acc, bal in subject_results:
        file.write(
            f"Subject {subject}: "
            f"Accuracy = {acc * 100:.2f}%, "
            f"Balanced Accuracy = {bal * 100:.2f}%\n"
        )

np.savez(
    OUTPUT_DIR / "eegnet_loso_predictions.npz",
    y_true=all_targets,
    y_pred=all_predictions
)


# -----------------------------
# 8. Confusion matrix plot
# -----------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")
plt.title("NEUROSYNC - EEGNet Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Class 0", "Class 1"])
plt.yticks([0, 1], ["Class 0", "Class 1"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "eegnet_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nResults saved successfully!")
print("Output:", results_path)
