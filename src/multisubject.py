
import re
import numpy as np
import mne

from pathlib import Path


# ==========================================
# NEUROSYNC - MULTI-SUBJECT EEG DATASET
# ==========================================

SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_multisubject():

    print("\nNEUROSYNC - MULTI-SUBJECT DATASET")
    print("=" * 50)

    print("Subjects:", SUBJECTS)
    print("Runs:", RUNS)

    # ==========================================
    # 1. DOWNLOAD DATA
    # ==========================================

    print("\nDownloading PhysioNet recordings...")

    files = mne.datasets.eegbci.load_data(
        subjects=SUBJECTS,
        runs=RUNS
    )

    all_epochs = []
    all_labels = []
    all_subjects = []
    all_runs = []

    channel_names = None
    sampling_rate = None

    # ==========================================
    # 2. PROCESS EACH RECORDING
    # ==========================================

    for index, file_path in enumerate(files, start=1):

        filename = Path(file_path).name

        match = re.search(
            r"S(\d+)R(\d+)\.edf",
            filename,
            re.IGNORECASE
        )

        if match is None:
            print("Skipping unrecognized file:", filename)
            continue

        subject_id = int(match.group(1))
        run_id = int(match.group(2))

        print(
            f"\n[{index}/{len(files)}] "
            f"Subject {subject_id}, Run {run_id}"
        )

        try:

            # Load EEG recording
            raw = mne.io.read_raw_edf(
                file_path,
                preload=True,
                verbose=False
            )

            mne.datasets.eegbci.standardize(raw)

            raw.pick("eeg")

            # Apply 1-40 Hz bandpass filter
            raw.filter(
                l_freq=1.0,
                h_freq=40.0,
                verbose=False
            )

            # Keep channel structure consistent
            if channel_names is None:
                channel_names = raw.ch_names.copy()
                sampling_rate = raw.info["sfreq"]

            elif raw.ch_names != channel_names:
                raise ValueError(
                    "Channel mismatch in " + filename
                )

            # ==================================
            # 3. EXTRACT EVENTS
            # ==================================

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

            # ==================================
            # 4. CREATE EPOCHS
            # ==================================

            epochs = mne.Epochs(
                raw,
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
                print("No valid epochs. Skipping.")
                continue

            all_epochs.append(X_subject)
            all_labels.append(y_subject)

            all_subjects.extend(
                [subject_id] * len(y_subject)
            )

            all_runs.extend(
                [run_id] * len(y_subject)
            )

            print("Epochs extracted:", len(X_subject))

        except Exception as error:

            print("Error processing", filename)
            print(error)

    # ==========================================
    # 5. COMBINE ALL SUBJECTS
    # ==========================================

    if not all_epochs:
        raise RuntimeError(
            "No EEG epochs were extracted."
        )

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    groups = np.asarray(all_subjects)
    run_ids = np.asarray(all_runs)

    # ==========================================
    # 6. DATASET SUMMARY
    # ==========================================

    print("\nMULTI-SUBJECT DATASET SUMMARY")
    print("=" * 50)

    print("Total epochs:", len(X))
    print("EEG shape:", X.shape)
    print("EEG channels:", X.shape[1])
    print("Sampling rate:", sampling_rate)

    print("\nClass distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Event {label}: {count} samples")

    print("\nSubject distribution:")

    for subject in np.unique(groups):

        count = np.sum(groups == subject)

        print(
            f"Subject {subject}: {count} epochs"
        )

    # ==========================================
    # 7. SAVE MULTI-SUBJECT DATASET
    # ==========================================

    output_path = OUTPUT_DIR / "multisubject_eeg.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups,
        run_ids=run_ids,
        channels=np.asarray(channel_names),
        sampling_rate=sampling_rate
    )

    print("\nDataset saved successfully!")
    print(output_path)

    print("\nNEUROSYNC multi-subject extraction completed!")


if __name__ == "__main__":
    run_multisubject()
