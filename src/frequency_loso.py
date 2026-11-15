
import re
import numpy as np
import mne
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
    confusion_matrix
)


# ==========================================
# NEUROSYNC - FREQUENCY BAND COMPARISON
# ==========================================

SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]

BANDS = {
    "Full EEG (1-40 Hz)": (1, 40),
    "Mu Alpha (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Mu Beta (8-30 Hz)": (8, 30)
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_model():

    return Pipeline([
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


def run_frequency_experiment():

    mne.set_log_level("ERROR")

    print("\nNEUROSYNC - FREQUENCY BAND ANALYSIS")
    print("=" * 55)

    # ==========================================
    # 1. DOWNLOAD / LOAD RECORDING PATHS
    # ==========================================

    print("\nLoading PhysioNet recording paths...")

    files = mne.datasets.eegbci.load_data(
        subjects=SUBJECTS,
        runs=RUNS
    )

    # Store data separately for each frequency band
    band_data = {
        name: {
            "X": [],
            "y": [],
            "groups": []
        }
        for name in BANDS
    }

    # ==========================================
    # 2. PROCESS RECORDINGS
    # ==========================================

    for index, file_path in enumerate(files, start=1):

        filename = Path(file_path).name

        match = re.search(
            r"S(\d+)R(\d+)\.edf",
            filename,
            re.IGNORECASE
        )

        if match is None:
            print("Skipping:", filename)
            continue

        subject_id = int(match.group(1))
        run_id = int(match.group(2))

        print(
            f"\n[{index}/{len(files)}] "
            f"Subject {subject_id}, Run {run_id}"
        )

        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False
        )

        mne.datasets.eegbci.standardize(raw)
        raw.pick("eeg")

        # Extract event information from original recording
        events, event_id = mne.events_from_annotations(
            raw,
            verbose=False
        )

        if "T1" not in event_id or "T2" not in event_id:
            print("T1/T2 events missing. Skipping.")
            continue

        selected_event_id = {
            "T1": event_id["T1"],
            "T2": event_id["T2"]
        }

        # ======================================
        # 3. FILTER CONTINUOUS EEG FOR EACH BAND
        # ======================================

        for band_name, (low, high) in BANDS.items():

            print("  Processing:", band_name)

            filtered_raw = raw.copy()

            filtered_raw.filter(
                l_freq=low,
                h_freq=high,
                verbose=False
            )

            # ==================================
            # 4. CREATE EPOCHS
            # ==================================

            epochs = mne.Epochs(
                filtered_raw,
                events,
                event_id=selected_event_id,
                tmin=0,
                tmax=4,
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                verbose=False
            )

            X_subject = epochs.get_data()
            y_subject = epochs.events[:, -1]

            if len(X_subject) == 0:
                print("  No valid epochs.")
                continue

            band_data[band_name]["X"].append(X_subject)
            band_data[band_name]["y"].append(y_subject)

            band_data[band_name]["groups"].extend(
                [subject_id] * len(y_subject)
            )

            print("  Extracted epochs:", len(X_subject))

            del filtered_raw
            del epochs

        del raw

    # ==========================================
    # 5. PREPARE LOSO VALIDATION
    # ==========================================

    print("\nFREQUENCY BAND LOSO EVALUATION")
    print("=" * 55)

    results = []

    cv = LeaveOneGroupOut()

    label_mapping = {
        2: 0,
        3: 1
    }

    # ==========================================
    # 6. TRAIN AND TEST EACH FREQUENCY BAND
    # ==========================================

    for band_name in BANDS:

        print("\nEvaluating:", band_name)

        X_parts = band_data[band_name]["X"]
        y_parts = band_data[band_name]["y"]
        groups_parts = band_data[band_name]["groups"]

        if not X_parts:
            print("No data available for this band.")
            continue

        X = np.concatenate(X_parts, axis=0)
        original_y = np.concatenate(y_parts, axis=0)

        groups = np.asarray(groups_parts)

        y = np.array([
            label_mapping[int(label)]
            for label in original_y
        ])

        print("EEG shape:", X.shape)
        print("Subjects:", len(np.unique(groups)))

        model = create_model()

        predictions = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            method="predict"
        )

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

        print(
            f"Accuracy: {accuracy * 100:.2f}%"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%"
        )

        print("Confusion Matrix:")
        print(matrix)

        results.append({
            "band": band_name,
            "accuracy": accuracy * 100,
            "balanced_accuracy": balanced_accuracy * 100,
            "samples": len(X)
        })

    # ==========================================
    # 7. PRINT COMPARISON TABLE
    # ==========================================

    print("\nFINAL FREQUENCY COMPARISON")
    print("=" * 65)

    print(
        f"{'Frequency Band':<25}"
        f"{'Accuracy':>15}"
        f"{'Balanced Acc.':>20}"
    )

    print("-" * 65)

    for result in results:

        print(
            f"{result['band']:<25}"
            f"{result['accuracy']:>14.2f}%"
            f"{result['balanced_accuracy']:>19.2f}%"
        )

    # ==========================================
    # 8. SAVE RESULTS
    # ==========================================

    result_file = OUTPUT_DIR / "frequency_loso_results.txt"

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("NEUROSYNC FREQUENCY BAND LOSO RESULTS\n")
        file.write("=" * 55 + "\n\n")

        for result in results:

            file.write(
                f"{result['band']}\n"
            )

            file.write(
                f"Samples: {result['samples']}\n"
            )

            file.write(
                f"Accuracy: {result['accuracy']:.2f}%\n"
            )

            file.write(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy']:.2f}%\n\n"
            )

    # ==========================================
    # 9. CREATE COMPARISON GRAPH
    # ==========================================

    if results:

        names = [
            result["band"]
            for result in results
        ]

        accuracies = [
            result["accuracy"]
            for result in results
        ]

        balanced = [
            result["balanced_accuracy"]
            for result in results
        ]

        x = np.arange(len(names))
        width = 0.35

        plt.figure(figsize=(12, 6))

        plt.bar(
            x - width / 2,
            accuracies,
            width,
            label="Accuracy"
        )

        plt.bar(
            x + width / 2,
            balanced,
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
            names,
            rotation=15,
            ha="right"
        )

        plt.ylabel("Performance (%)")
        plt.title("NEUROSYNC - Frequency Band Comparison")

        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "frequency_band_comparison.png",
            dpi=300
        )

        plt.close()

    print("\nGenerated files:")
    print("1. outputs/frequency_loso_results.txt")
    print("2. outputs/frequency_band_comparison.png")

    print("\nFrequency band experiment completed!")


if __name__ == "__main__":
    run_frequency_experiment()
