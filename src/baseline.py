
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from joblib import dump

from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


def run_baseline():

    print("\nNEUROSYNC - CSP + LDA CLASSIFICATION")
    print("=" * 50)

    # ==========================================
    # 1. LOAD PROCESSED EEG DATA
    # ==========================================

    dataset_path = Path("data/processed/eeg_dataset.npz")

    if not dataset_path.exists():
        raise FileNotFoundError(
            "EEG dataset not found. Run run_physionet.bat first."
        )

    dataset = np.load(dataset_path)

    X = dataset["X"]
    original_labels = dataset["y"]

    print("\nDataset loaded successfully!")
    print("EEG shape:", X.shape)

    # ==========================================
    # 2. CONVERT LABELS TO BINARY FORMAT
    # ==========================================

    # Original event codes: 2 and 3
    # Machine learning labels: 0 and 1

    label_mapping = {
        2: 0,
        3: 1
    }

    y = np.array([
        label_mapping[int(label)]
        for label in original_labels
    ])

    print("\nClass distribution:")

    unique, counts = np.unique(y, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples")

    # ==========================================
    # 3. CREATE CSP + LDA PIPELINE
    # ==========================================

    csp = CSP(
    n_components=4,
    reg=0.1,
    log=True,
    norm_trace=False
)

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    )

    model = Pipeline([
        ("csp", csp),
        ("lda", lda)
    ])

    # ==========================================
    # 4. STRATIFIED 5-FOLD CROSS-VALIDATION
    # ==========================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\nRunning 5-fold cross-validation...")

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    # ==========================================
    # 5. EVALUATE MODEL PERFORMANCE
    # ==========================================

    accuracy = accuracy_score(y, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print("\nMODEL PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["T1 event", "T2 event"],
            zero_division=0
        )
    )

    # ==========================================
    # 6. SAVE CONFUSION MATRIX VISUALIZATION
    # ==========================================

    output_dir = Path("outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix, interpolation="nearest")

    plt.title("NEUROSYNC - CSP + LDA Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["T1", "T2"])
    plt.yticks([0, 1], ["T1", "T2"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        output_dir / "csp_lda_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # 7. TRAIN FINAL MODEL ON ALL AVAILABLE DATA
    # ==========================================

    print("\nTraining final model on all 45 epochs...")

    model.fit(X, y)

    model_dir = Path("models")
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dump(
        model,
        model_dir / "csp_lda_model.joblib"
    )

    # ==========================================
    # 8. SAVE EXPERIMENT RESULTS
    # ==========================================

    with open(
        output_dir / "baseline_results.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("NEUROSYNC - CSP + LDA RESULTS\n")
        file.write("=" * 40 + "\n")

        file.write(f"Dataset shape: {X.shape}\n")
        file.write(f"Number of epochs: {len(X)}\n")
        file.write(f"Number of channels: {X.shape[1]}\n")

        file.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Balanced accuracy: "
            f"{balanced_accuracy * 100:.2f}%\n"
        )

        file.write("\nConfusion Matrix:\n")
        file.write(str(matrix))

        file.write("\n\nClassification Report:\n")

        file.write(
            classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["T1 event", "T2 event"],
                zero_division=0
            )
        )

    # ==========================================
    # 9. FINAL OUTPUT
    # ==========================================

    print("\nGENERATED FILES")
    print("=" * 40)

    print("1. outputs/csp_lda_confusion_matrix.png")
    print("2. outputs/baseline_results.txt")
    print("3. models/csp_lda_model.joblib")

    print("\nNEUROSYNC baseline experiment completed!")


if __name__ == "__main__":
    run_baseline()

