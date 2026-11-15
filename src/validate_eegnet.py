
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


# -------------------------
# Configuration
# -------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "multisubject_eeg.npz"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33]
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

torch.set_num_threads(4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------
# EEGNet
# -------------------------

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                8, 16, (n_channels, 1),
                groups=8, bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                16, 16, (1, 16),
                padding=(0, 8), groups=16, bias=False
            ),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            size = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(size, n_classes)

    def features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.separable(x)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


def normalize_from_train(X_train, *other_sets):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)

    normalized = [(X_train - mean) / std]

    for X_other in other_sets:
        normalized.append((X_other - mean) / std)

    return normalized


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


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()


def validation_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            loss = criterion(model(xb), yb)
            total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def select_epoch_count(X_train, y_train, groups_train, seed):
    """Select epoch count using only the outer training subjects."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.22,
        random_state=2026
    )

    train_idx, val_idx = next(
        splitter.split(X_train, y_train, groups_train)
    )

    X_inner_train = X_train[train_idx]
    y_inner_train = y_train[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    X_inner_train, X_val = normalize_from_train(
        X_inner_train, X_val
    )

    train_loader = make_loader(
        X_inner_train, y_inner_train, True
    )
    val_loader = make_loader(
        X_val, y_val, False
    )

    set_seed(seed)

    model = EEGNet(
        X_train.shape[1],
        X_train.shape[2]
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
        train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        loss = validation_loss(
            model, val_loader, criterion
        )

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_weights)

    return best_epoch


def fit_final_eegnet(X_train, y_train, X_test, epochs, seed):
    """Retrain on all outer training subjects."""

    X_train_scaled, X_test_scaled = normalize_from_train(
        X_train, X_test
    )

    train_loader = make_loader(
        X_train_scaled, y_train, True
    )

    set_seed(seed)

    model = EEGNet(
        X_train.shape[1],
        X_train.shape[2]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train_one_epoch(
            model, train_loader, optimizer, criterion
        )

    model.eval()

    test_tensor = torch.tensor(
        X_test_scaled[:, np.newaxis, :, :],
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        predictions = model(test_tensor).argmax(1).cpu().numpy()

    return predictions


# -------------------------
# Load data
# -------------------------

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"]

y = np.where(y == 2, 0, 1).astype(np.int64)

print("Dataset:", X.shape)
print("Subjects:", np.unique(groups))
print("EEGNet seeds:", SEEDS)


# -------------------------
# Identical outer LOSO folds
# -------------------------

logo = LeaveOneGroupOut()

csp_predictions = np.zeros(len(y), dtype=np.int64)
seed_predictions = {
    seed: np.zeros(len(y), dtype=np.int64)
    for seed in SEEDS
}

fold_details = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):

    held_out_subject = groups[test_idx][0]

    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    print("\n" + "=" * 55)
    print(f"Fold {fold}/10 | Held-out subject: {held_out_subject}")
    print("=" * 55)

    # CSP + LDA: train on all nine outer training subjects
    csp = CSP(
        n_components=4,
        reg=None,
        log=True,
        norm_trace=False
    )

    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_csp, y_train)

    csp_fold_pred = lda.predict(X_test_csp)
    csp_predictions[test_idx] = csp_fold_pred

    print(
        "CSP + LDA accuracy:",
        f"{accuracy_score(y_test, csp_fold_pred) * 100:.2f}%"
    )

    # EEGNet: repeat with multiple initialization/data-order seeds
    for seed in SEEDS:

        selected_epochs = select_epoch_count(
            X_train,
            y_train,
            groups_train,
            seed
        )

        predictions = fit_final_eegnet(
            X_train,
            y_train,
            X_test,
            selected_epochs,
            seed
        )

        seed_predictions[seed][test_idx] = predictions

        fold_acc = accuracy_score(y_test, predictions)
        fold_bal = balanced_accuracy_score(y_test, predictions)

        print(
            f"EEGNet seed {seed} | "
            f"selected epochs: {selected_epochs} | "
            f"accuracy: {fold_acc * 100:.2f}% | "
            f"balanced: {fold_bal * 100:.2f}%"
        )

        fold_details.append(
            (held_out_subject, seed, selected_epochs,
             fold_acc, fold_bal)
        )


# -------------------------
# Aggregate results
# -------------------------

csp_acc = accuracy_score(y, csp_predictions)
csp_bal = balanced_accuracy_score(y, csp_predictions)

print("\n" + "=" * 55)
print("CONTROLLED LOSO RESULTS")
print("=" * 55)

print(
    f"CSP + LDA | Accuracy: {csp_acc * 100:.2f}% | "
    f"Balanced accuracy: {csp_bal * 100:.2f}%"
)

seed_scores = []

for seed in SEEDS:
    pred = seed_predictions[seed]

    acc = accuracy_score(y, pred)
    bal = balanced_accuracy_score(y, pred)

    seed_scores.append((acc, bal))

    print(
        f"EEGNet seed {seed} | "
        f"Accuracy: {acc * 100:.2f}% | "
        f"Balanced accuracy: {bal * 100:.2f}%"
    )

accuracies = np.array([x[0] for x in seed_scores])
balanced_scores = np.array([x[1] for x in seed_scores])

print("\nEEGNet across seeds:")
print(
    f"Accuracy: {accuracies.mean() * 100:.2f}% "
    f"+/- {accuracies.std(ddof=1) * 100:.2f}%"
)
print(
    f"Balanced accuracy: {balanced_scores.mean() * 100:.2f}% "
    f"+/- {balanced_scores.std(ddof=1) * 100:.2f}%"
)

# Average the three seed probabilities is not possible from
# class labels alone. Majority vote gives a simple ensemble summary.
stacked = np.stack(
    [seed_predictions[s] for s in SEEDS],
    axis=0
)

ensemble_predictions = (
    np.sum(stacked, axis=0) >= 2
).astype(np.int64)

ensemble_acc = accuracy_score(y, ensemble_predictions)
ensemble_bal = balanced_accuracy_score(y, ensemble_predictions)

print(
    f"\nEEGNet majority vote | Accuracy: {ensemble_acc * 100:.2f}% | "
    f"Balanced accuracy: {ensemble_bal * 100:.2f}%"
)

print("\nCSP + LDA confusion matrix:")
print(confusion_matrix(y, csp_predictions))

for seed in SEEDS:
    print(f"\nEEGNet seed {seed} confusion matrix:")
    print(confusion_matrix(y, seed_predictions[seed]))
# Save selected epoch counts for final model training
epoch_counts = np.array(
    [item[2] for item in fold_details],
    dtype=np.int64
)

median_epochs = int(np.median(epoch_counts))

print("\nSelected epoch counts:", epoch_counts.tolist())
print("Median selected epochs:", median_epochs)

np.save(
    OUTPUT_DIR / "selected_epoch_counts.npy",
    epoch_counts
)

with open(OUTPUT_DIR / "selected_epoch_summary.txt", "w") as f:
    f.write(f"Selected epoch counts: {epoch_counts.tolist()}\n")
    f.write(f"Median selected epochs: {median_epochs}\n")
# Save everything
np.savez(
    OUTPUT_DIR / "controlled_validation_predictions.npz",
    y_true=y,
    csp_lda=csp_predictions,
    **{
        f"eegnet_seed_{seed}": seed_predictions[seed]
        for seed in SEEDS
    },
    eegnet_majority_vote=ensemble_predictions
)

report_path = OUTPUT_DIR / "controlled_validation_results.txt"

with open(report_path, "w") as f:
    f.write("NEUROSYNC - Controlled LOSO Validation\n\n")
    f.write(f"Subjects: {len(np.unique(groups))}\n")
    f.write(f"EEGNet seeds: {SEEDS}\n\n")

    f.write(
        f"CSP + LDA accuracy: {csp_acc * 100:.2f}%\n"
    )
    f.write(
        f"CSP + LDA balanced accuracy: {csp_bal * 100:.2f}%\n\n"
    )

    for seed, (acc, bal) in zip(SEEDS, seed_scores):
        f.write(
            f"EEGNet seed {seed}: "
            f"accuracy={acc * 100:.2f}%, "
            f"balanced_accuracy={bal * 100:.2f}%\n"
        )

    f.write(
        f"\nEEGNet mean accuracy: {accuracies.mean() * 100:.2f}%\n"
    )
    f.write(
        f"EEGNet accuracy SD: {accuracies.std(ddof=1) * 100:.2f}%\n"
    )
    f.write(
        f"EEGNet mean balanced accuracy: "
        f"{balanced_scores.mean() * 100:.2f}%\n"
    )
    f.write(
        f"EEGNet balanced accuracy SD: "
        f"{balanced_scores.std(ddof=1) * 100:.2f}%\n"
    )
    f.write(
        f"\nMajority vote accuracy: {ensemble_acc * 100:.2f}%\n"
    )
    f.write(
        f"Majority vote balanced accuracy: "
        f"{ensemble_bal * 100:.2f}%\n"
    )

print("\nSaved:", report_path)
print("Controlled validation completed.")
