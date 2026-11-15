
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


def run_loso():

    print("\nNEUROSYNC - LEAVE ONE SUBJECT OUT")
    print("=" * 50)

    # ==========================================
    # 1. LOAD MULTI-SUBJECT DATASET
    # ==========================================

    dataset_path = Path(
        "data/processed/multisubject_eeg.npz"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Multi-subject EEG dataset not found."
        )

    dataset = np.load(dataset_path)

    X = dataset["X"]
    original_labels = dataset["y"]
    groups = dataset["groups"]

    print("\nDataset loaded!")
    print("EEG shape:", X.shape)
    print("Subjects:", np.unique(groups))

    # ==========================================
    # 2. CONVERT LABELS
    # ==========================================

    label_mapping = {
        2: 0,
        3: 1
    }

    y = np.array([
        label_mapping[int(label)]
        for label in original_labels
    ])

    print("\nClass distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    # ==========================================
    # 3. CREATE CSP + LDA PIPELINE
    # ==========================================

    model = Pipeline([
        (
            "csp",
            CSP(
                n_components=4,
                reg=0.1,
                log=True,
                norm_trace=False
            )
        ),
        (
            "lda",
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            )
        )
    ])

    # ==========================================
    # 4. LEAVE-ONE-SUBJECT-OUT VALIDATION
    # ==========================================

    cv = LeaveOneGroupOut()

    print("\nStarting LOSO cross-validation...")
    print("Total subjects:", len(np.unique(groups)))

    predictions = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict"
    )

    # ==========================================
    # 5. OVERALL PERFORMANCE
    # ==========================================

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print("\nOVERALL LOSO PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["T1 event", "T2 event"],
        zero_division=0
    )

    print(report)

    # ==========================================
    # 6. INDIVIDUAL SUBJECT PERFORMANCE
    # ==========================================

    print("\nSUBJECT-WISE PERFORMANCE")
    print("=" * 50)

    subject_results = []

    for subject in np.unique(groups):

        mask = groups == subject

        subject_accuracy = accuracy_score(
            y[mask],
            predictions[mask]
        )

        subject_balanced_accuracy = balanced_accuracy_score(
            y[mask],
            predictions[mask]
        )

        subject_results.append([
            int(subject),
            int(np.sum(mask)),
            subject_accuracy,
            subject_balanced_accuracy
        ])

        print(
            f"Subject {subject}: "
            f"Accuracy = {subject_accuracy * 100:.2f}% | "
            f"Balanced Accuracy = "
            f"{subject_balanced_accuracy * 100:.2f}%"
        )

    # ==========================================
    # 7. SAVE RESULTS
    # ==========================================

    output_dir = Path("outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_dir / "loso_results.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "NEUROSYNC - LEAVE ONE SUBJECT OUT RESULTS\n"
        )

        file.write("=" * 50 + "\n")

        file.write(f"Dataset shape: {X.shape}\n")
        file.write(f"Subjects: {len(np.unique(groups))}\n")

        file.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%\n"
        )

        file.write("\nConfusion Matrix:\n")
        file.write(str(matrix))

        file.write("\n\nClassification Report:\n")
        file.write(report)

        file.write("\n\nSubject-wise Results:\n")

        for subject, samples, acc, bal_acc in subject_results:

            file.write(
                f"Subject {subject}: "
                f"Samples={samples}, "
                f"Accuracy={acc * 100:.2f}%, "
                f"Balanced Accuracy={bal_acc * 100:.2f}%\n"
            )

    # ==========================================
    # 8. CONFUSION MATRIX VISUALIZATION
    # ==========================================

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix, interpolation="nearest")

    plt.title("NEUROSYNC - LOSO Confusion Matrix")
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
        output_dir / "loso_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # 9. SUBJECT ACCURACY GRAPH
    # ==========================================

    subject_ids = [
        result[0] for result in subject_results
    ]

    subject_accuracies = [
        result[2] * 100 for result in subject_results
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        subject_ids,
        subject_accuracies
    )

    plt.axhline(
        50,
        linestyle="--",
        label="50% reference"
    )

    plt.xlabel("Subject ID")
    plt.ylabel("Accuracy (%)")
    plt.title("NEUROSYNC - Subject-wise LOSO Accuracy")

    plt.xticks(subject_ids)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "loso_subject_accuracy.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # 10. FINAL OUTPUT
    # ==========================================

    print("\nGENERATED FILES")
    print("=" * 50)

    print("1. outputs/loso_results.txt")
    print("2. outputs/loso_confusion_matrix.png")
    print("3. outputs/loso_subject_accuracy.png")

    print("\nNEUROSYNC LOSO experiment completed!")


if __name__ == "__main__":
    run_loso()
