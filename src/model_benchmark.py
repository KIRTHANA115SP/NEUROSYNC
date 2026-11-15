
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# NEUROSYNC - CLASSICAL MODEL BENCHMARK
# ==========================================

DATASET_PATH = Path(
    "data/processed/multisubject_eeg.npz"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_models():

    # CSP extracts spatial EEG features.
    # It is fitted independently inside each CV training fold.

    lda_model = Pipeline([
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
            "classifier",
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            )
        )
    ])

    svm_model = Pipeline([
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
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            )
        )
    ])

    rf_model = Pipeline([
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
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )
        )
    ])

    return {
        "CSP + LDA": lda_model,
        "CSP + SVM": svm_model,
        "CSP + Random Forest": rf_model
    }


def run_benchmark():

    print("\nNEUROSYNC - CLASSICAL ML BENCHMARK")
    print("=" * 55)

    # ==========================================
    # 1. LOAD DATASET
    # ==========================================

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Multi-subject dataset not found. "
            "Run multisubject.py first."
        )

    dataset = np.load(DATASET_PATH)

    X = dataset["X"]
    original_labels = dataset["y"]
    groups = dataset["groups"]

    print("\nDataset loaded successfully!")
    print("EEG shape:", X.shape)
    print("Number of subjects:", len(np.unique(groups)))

    # ==========================================
    # 2. CONVERT EVENT LABELS
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
        print(f"Class {label}: {count} samples")

    # ==========================================
    # 3. PREPARE LOSO VALIDATION
    # ==========================================

    cv = LeaveOneGroupOut()

    models = create_models()

    results = []
    all_predictions = {}

    # ==========================================
    # 4. EVALUATE EACH MODEL
    # ==========================================

    for model_name, model in models.items():

        print("\n" + "=" * 55)
        print("Evaluating:", model_name)
        print("=" * 55)

        predictions = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            method="predict",
            n_jobs=1
        )

        all_predictions[model_name] = predictions

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

        print(f"\nAccuracy: {accuracy * 100:.2f}%")

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

        # ======================================
        # 5. SUBJECT-WISE PERFORMANCE
        # ======================================

        subject_results = []

        print("\nSubject-wise accuracy:")

        for subject in np.unique(groups):

            mask = groups == subject

            subject_accuracy = accuracy_score(
                y[mask],
                predictions[mask]
            )

            subject_balanced = balanced_accuracy_score(
                y[mask],
                predictions[mask]
            )

            subject_results.append({
                "subject": int(subject),
                "accuracy": subject_accuracy * 100,
                "balanced_accuracy": subject_balanced * 100
            })

            print(
                f"Subject {subject}: "
                f"{subject_accuracy * 100:.2f}% "
                f"(Balanced: {subject_balanced * 100:.2f}%)"
            )

        results.append({
            "model": model_name,
            "accuracy": accuracy * 100,
            "balanced_accuracy": balanced_accuracy * 100,
            "confusion_matrix": matrix,
            "report": report,
            "subject_results": subject_results
        })

    # ==========================================
    # 6. FINAL COMPARISON
    # ==========================================

    print("\nFINAL MODEL COMPARISON")
    print("=" * 75)

    print(
        f"{'Model':<25}"
        f"{'Accuracy':>15}"
        f"{'Balanced Accuracy':>25}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['model']:<25}"
            f"{result['accuracy']:>14.2f}%"
            f"{result['balanced_accuracy']:>24.2f}%"
        )

    # ==========================================
    # 7. SAVE TEXT RESULTS
    # ==========================================

    result_path = OUTPUT_DIR / "model_benchmark_results.txt"

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("NEUROSYNC - CLASSICAL ML BENCHMARK\n")
        file.write("=" * 55 + "\n\n")

        file.write(f"Dataset shape: {X.shape}\n")
        file.write(f"Subjects: {len(np.unique(groups))}\n\n")

        for result in results:

            file.write("\n" + "=" * 45 + "\n")
            file.write(result["model"] + "\n")
            file.write("=" * 45 + "\n")

            file.write(
                f"Accuracy: {result['accuracy']:.2f}%\n"
            )

            file.write(
                f"Balanced accuracy: "
                f"{result['balanced_accuracy']:.2f}%\n"
            )

            file.write(
                "\nConfusion matrix:\n"
                + str(result["confusion_matrix"])
                + "\n"
            )

            file.write(
                "\nClassification report:\n"
                + result["report"]
                + "\n"
            )

            file.write("\nSubject-wise results:\n")

            for subject_result in result["subject_results"]:

                file.write(
                    f"Subject {subject_result['subject']}: "
                    f"Accuracy = "
                    f"{subject_result['accuracy']:.2f}%, "
                    f"Balanced accuracy = "
                    f"{subject_result['balanced_accuracy']:.2f}%\n"
                )

    # ==========================================
    # 8. MODEL ACCURACY COMPARISON GRAPH
    # ==========================================

    model_names = [
        result["model"]
        for result in results
    ]

    accuracies = [
        result["accuracy"]
        for result in results
    ]

    balanced_accuracies = [
        result["balanced_accuracy"]
        for result in results
    ]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy"
    )

    plt.bar(
        x + width / 2,
        balanced_accuracies,
        width,
        label="Balanced Accuracy"
    )

    plt.axhline(
        50,
        linestyle="--",
        label="50% reference"
    )

    plt.xticks(
        x,
        model_names,
        rotation=10
    )

    plt.ylabel("Performance (%)")
    plt.title("NEUROSYNC - Classical Model Comparison")

    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_benchmark_comparison.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # 9. FINAL OUTPUT
    # ==========================================

    print("\nGENERATED FILES")
    print("=" * 50)

    print("1. outputs/model_benchmark_results.txt")
    print("2. outputs/model_benchmark_comparison.png")

    print("\nNEUROSYNC benchmark completed!")


if __name__ == "__main__":
    run_benchmark()
