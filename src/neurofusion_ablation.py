
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "multisubject_eeg.npz"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "ablation_checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33]

SAMPLE_RATE = 160.0
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 32

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

EXPERIMENTS = {
    "spatial": ("spatial",),
    "temporal": ("temporal",),
    "frequency": ("frequency",),
    "spatial_temporal": ("spatial", "temporal"),
    "spatial_frequency": ("spatial", "frequency"),
    "temporal_frequency": ("temporal", "frequency"),
    "full": ("spatial", "temporal", "frequency")
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================================================
# 2. NEUROFUSION-T MODEL
# =====================================================

class NeuroFusionAblation(nn.Module):

    def __init__(
        self,
        n_channels,
        n_times,
        enabled_branches,
        sample_rate=160.0
    ):
        super().__init__()

        self.enabled = enabled_branches
        self.sample_rate = sample_rate

        if "spatial" in self.enabled:
            self.spatial_branch = nn.Sequential(
                nn.Conv2d(
                    1, 16,
                    kernel_size=(n_channels, 1),
                    bias=False
                ),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 64)
            )

        if "temporal" in self.enabled:
            self.temporal_branch = nn.Sequential(
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
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )

        if "frequency" in self.enabled:
            self.frequency_branch = nn.Sequential(
                nn.Linear(n_channels * 2, 128),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LayerNorm(64)
            )

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

    def get_frequency_features(self, eeg):

        n_times = eeg.shape[-1]

        spectrum = torch.fft.rfft(eeg, dim=-1)
        power = spectrum.abs().pow(2) / n_times

        frequencies = torch.fft.rfftfreq(
            n_times,
            d=1.0 / self.sample_rate
        ).to(eeg.device)

        mu_mask = (
            (frequencies >= 8.0) &
            (frequencies < 13.0)
        )

        beta_mask = (
            (frequencies >= 13.0) &
            (frequencies <= 30.0)
        )

        mu_power = power[:, :, mu_mask].mean(dim=-1)
        beta_power = power[:, :, beta_mask].mean(dim=-1)

        mu_power = torch.log(mu_power + 1e-10)
        beta_power = torch.log(beta_power + 1e-10)

        return torch.cat(
            [mu_power, beta_power],
            dim=1
        )

    def forward(self, x):

        eeg = x.squeeze(1)
        tokens = []

        if "spatial" in self.enabled:
            spatial = self.spatial_branch(x)
            tokens.append(spatial)

        if "temporal" in self.enabled:
            temporal = self.temporal_branch(eeg)
            tokens.append(temporal)

        if "frequency" in self.enabled:
            frequency = self.get_frequency_features(eeg)
            frequency = self.frequency_branch(frequency)
            tokens.append(frequency)

        tokens = torch.stack(tokens, dim=1)

        # Cross-branch attention is meaningful when
        # more than one branch is enabled.
        if len(self.enabled) > 1:
            attended, _ = self.attention(
                tokens,
                tokens,
                tokens,
                need_weights=False
            )

            tokens = self.fusion_norm(tokens + attended)

        fused = tokens.mean(dim=1)

        return self.classifier(fused)


# =====================================================
# 3. DATA HELPERS
# =====================================================

def normalize_from_train(X_train, *other_sets):

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)

    std = np.maximum(std, 1e-8)

    results = [(X_train - mean) / std]

    for item in other_sets:
        results.append((item - mean) / std)

    return results


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


# =====================================================
# 4. TRAINING HELPERS
# =====================================================

def train_epoch(model, loader, optimizer, criterion):

    model.train()

    for xb, yb in loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        loss = criterion(model(xb), yb)

        loss.backward()
        optimizer.step()


def get_validation_loss(model, loader, criterion):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            loss = criterion(model(xb), yb)

            total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def select_epochs(
    X_train,
    y_train,
    groups_train,
    branches,
    seed
):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.22,
        random_state=2026
    )

    train_idx, val_idx = next(
        splitter.split(
            X_train,
            y_train,
            groups_train
        )
    )

    X_inner = X_train[train_idx]
    y_inner = y_train[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    X_inner, X_val = normalize_from_train(
        X_inner,
        X_val
    )

    train_loader = make_loader(
        X_inner,
        y_inner,
        True
    )

    val_loader = make_loader(
        X_val,
        y_val,
        False
    )

    set_seed(seed)

    model = NeuroFusionAblation(
        X_train.shape[1],
        X_train.shape[2],
        branches,
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
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss = get_validation_loss(
            model,
            val_loader,
            criterion
        )

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0

        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_weights)

    return best_epoch


def train_final_model(
    X_train,
    y_train,
    X_test,
    branches,
    epochs,
    seed
):

    X_train_scaled, X_test_scaled = normalize_from_train(
        X_train,
        X_test
    )

    train_loader = make_loader(
        X_train_scaled,
        y_train,
        True
    )

    set_seed(seed)

    model = NeuroFusionAblation(
        X_train.shape[1],
        X_train.shape[2],
        branches,
        SAMPLE_RATE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):

        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

    model.eval()

    test_tensor = torch.tensor(
        X_test_scaled[:, np.newaxis, :, :],
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        predictions = (
            model(test_tensor)
            .argmax(1)
            .cpu()
            .numpy()
        )

    return predictions


# =====================================================
# 5. INTERRUPTION-SAFE CHECKPOINTS
# =====================================================

def checkpoint_path(experiment, seed, subject):

    return CHECKPOINT_DIR / (
        f"{experiment}_seed{seed}_subject{subject}.npz"
    )


def save_checkpoint(
    experiment,
    seed,
    subject,
    test_idx,
    predictions,
    y_test,
    epochs
):

    path = checkpoint_path(
        experiment,
        seed,
        subject
    )

    temp_path = path.with_name(path.stem + ".tmp.npz")

    # Save to a temporary file first. Replace the final
    # checkpoint only after the write completes.
    with open(temp_path, "wb") as file:

        np.savez_compressed(
            file,
            experiment=np.array(experiment),
            seed=np.array(seed),
            subject=np.array(str(subject)),
            test_idx=test_idx,
            predictions=predictions,
            y_true=y_test,
            epochs=np.array(epochs)
        )

    os.replace(temp_path, path)


def load_checkpoint(
    experiment,
    seed,
    subject,
    expected_idx,
    expected_y
):

    path = checkpoint_path(
        experiment,
        seed,
        subject
    )

    if not path.exists():
        return None

    try:

        with np.load(path, allow_pickle=False) as saved:

            indices = saved["test_idx"]
            predictions = saved["predictions"]
            saved_y = saved["y_true"]

            if not np.array_equal(indices, expected_idx):
                return None

            if not np.array_equal(saved_y, expected_y):
                return None

            if predictions.shape != expected_y.shape:
                return None

            return {
                "predictions": predictions.copy(),
                "epochs": int(saved["epochs"])
            }

    except Exception as error:

        print(
            f"Invalid checkpoint {path.name}: {error}"
        )

        return None


# =====================================================
# 6. FINAL REPORT GENERATION
# =====================================================

def generate_report(X, y, groups):

    report_path = OUTPUT_DIR / "neurofusion_ablation_results.txt"

    logo = LeaveOneGroupOut()

    all_results = {}
    prediction_arrays = {}

    with open(report_path, "w") as report:

        report.write(
            "NEUROSYNC - NeuroFusion-T Ablation Study\n"
        )

        report.write("=" * 60 + "\n\n")

        for experiment, branches in EXPERIMENTS.items():

            print("\n" + "#" * 60)
            print("EXPERIMENT:", experiment.upper())
            print("#" * 60)

            seed_predictions = {}

            for seed in SEEDS:

                predictions = np.full(
                    len(y),
                    -1,
                    dtype=np.int64
                )

                completed = True

                for train_idx, test_idx in logo.split(
                    X, y, groups
                ):

                    subject = groups[test_idx][0]

                    saved = load_checkpoint(
                        experiment,
                        seed,
                        subject,
                        test_idx,
                        y[test_idx]
                    )

                    if saved is None:
                        completed = False
                        break

                    predictions[test_idx] = saved["predictions"]

                if completed:

                    seed_predictions[seed] = predictions

                else:

                    print(
                        f"Still incomplete: {experiment}, "
                        f"seed {seed}"
                    )

            if len(seed_predictions) != len(SEEDS):

                print(
                    f"Report waiting for all runs: {experiment}"
                )

                continue

            scores = []

            for seed in SEEDS:

                pred = seed_predictions[seed]

                acc = accuracy_score(y, pred)
                bal = balanced_accuracy_score(y, pred)

                scores.append((acc, bal))

                print(
                    f"{experiment} | Seed {seed} | "
                    f"Accuracy: {acc * 100:.2f}% | "
                    f"Balanced: {bal * 100:.2f}%"
                )

                prediction_arrays[
                    f"{experiment}_seed_{seed}"
                ] = pred

            acc_values = np.array([s[0] for s in scores])
            bal_values = np.array([s[1] for s in scores])

            stacked = np.stack(
                [seed_predictions[s] for s in SEEDS]
            )

            majority = (
                np.sum(stacked, axis=0) >= 2
            ).astype(np.int64)

            majority_acc = accuracy_score(y, majority)
            majority_bal = balanced_accuracy_score(y, majority)

            result = {
                "accuracy_mean": float(acc_values.mean()),
                "accuracy_std": float(acc_values.std(ddof=1)),
                "balanced_mean": float(bal_values.mean()),
                "balanced_std": float(bal_values.std(ddof=1)),
                "majority_accuracy": float(majority_acc),
                "majority_balanced": float(majority_bal)
            }

            all_results[experiment] = result

            prediction_arrays[
                f"{experiment}_majority"
            ] = majority

            report.write(f"Experiment: {experiment}\n")

            report.write(
                f"Mean accuracy: {result['accuracy_mean'] * 100:.2f}% "
                f"+/- {result['accuracy_std'] * 100:.2f}%\n"
            )

            report.write(
                f"Mean balanced accuracy: "
                f"{result['balanced_mean'] * 100:.2f}% "
                f"+/- {result['balanced_std'] * 100:.2f}%\n"
            )

            report.write(
                f"Majority vote accuracy: "
                f"{majority_acc * 100:.2f}%\n"
            )

            report.write(
                f"Majority vote balanced accuracy: "
                f"{majority_bal * 100:.2f}%\n\n"
            )

            report.flush()

    if prediction_arrays:

        np.savez_compressed(
            OUTPUT_DIR / "neurofusion_ablation_predictions.npz",
            y_true=y,
            **prediction_arrays
        )

    print("\nCurrent report:", report_path)

    return all_results


# =====================================================
# 7. MAIN EXPERIMENT
# =====================================================

def main():

    data = np.load(DATA_PATH)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["groups"]

    y = np.where(y == 2, 0, 1).astype(np.int64)

    print("\nNEUROSYNC - Resumable Ablation Study")
    print("Device:", DEVICE)
    print("Dataset:", X.shape)
    print("Subjects:", np.unique(groups))
    print("Seeds:", SEEDS)
    print("Checkpoint folder:", CHECKPOINT_DIR)

    logo = LeaveOneGroupOut()

    for experiment, branches in EXPERIMENTS.items():

        print("\n" + "=" * 60)
        print("EXPERIMENT:", experiment.upper())
        print("BRANCHES:", branches)
        print("=" * 60)

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

            for seed in SEEDS:

                existing = load_checkpoint(
                    experiment,
                    seed,
                    subject,
                    test_idx,
                    y_test
                )

                if existing is not None:

                    print(
                        f"SKIPPED | {experiment} | "
                        f"Subject {subject} | Seed {seed} "
                        f"| Already saved"
                    )

                    continue

                print(
                    f"\nTraining | {experiment} | "
                    f"Fold {fold}/10 | Subject {subject} | "
                    f"Seed {seed}"
                )

                epochs = select_epochs(
                    X_train,
                    y_train,
                    groups_train,
                    branches,
                    seed
                )

                predictions = train_final_model(
                    X_train,
                    y_train,
                    X_test,
                    branches,
                    epochs,
                    seed
                )

                acc = accuracy_score(y_test, predictions)
                bal = balanced_accuracy_score(y_test, predictions)

                save_checkpoint(
                    experiment,
                    seed,
                    subject,
                    test_idx,
                    predictions,
                    y_test,
                    epochs
                )

                print(
                    f"SAVED | Epochs {epochs} | "
                    f"Accuracy {acc * 100:.2f}% | "
                    f"Balanced {bal * 100:.2f}%"
                )

        # Rebuild the report after every configuration.
        generate_report(X, y, groups)

    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENT FINISHED")
    print("=" * 60)

    generate_report(X, y, groups)

    print("\nAll available results saved successfully.")


if __name__ == "__main__":
    main()
