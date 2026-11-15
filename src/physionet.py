
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne


def run_physionet():

    print("\nNEUROSYNC - PhysioNet EEG Exploration")
    print("=" * 45)

    # ==========================================
    # 1. DOWNLOAD PHYSIONET EEG DATA
    # ==========================================

    print("\nDownloading EEG data...")

    files = mne.datasets.eegbci.load_data(
        subjects=[1],
        runs=[4, 8, 12]
    )

    raw_list = []

    for file in files:
        raw = mne.io.read_raw_edf(
            file,
            preload=True,
            verbose=False
        )

        mne.datasets.eegbci.standardize(raw)
        raw_list.append(raw)

    # ==========================================
    # 2. COMBINE EEG RECORDINGS
    # ==========================================

    raw = mne.concatenate_raws(raw_list)

    raw.pick("eeg")

    raw.set_montage(
        "standard_1005",
        on_missing="ignore"
    )

    print("Sampling frequency:", raw.info["sfreq"])
    print("EEG channels:", len(raw.ch_names))

    # ==========================================
    # 3. FILTER EEG SIGNAL
    # ==========================================

    print("\nFiltering EEG signals...")

    raw.filter(
        l_freq=1.0,
        h_freq=40.0,
        verbose=False
    )

    # ==========================================
    # 4. CREATE OUTPUT DIRECTORIES
    # ==========================================

    output_dir = Path("outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # 5. VISUALIZE EEG WAVEFORMS
    # ==========================================

    print("\nGenerating EEG waveform plot...")

    data, times = raw.get_data(
        picks="eeg",
        return_times=True
    )

    number_of_channels = min(8, len(raw.ch_names))

    plt.figure(figsize=(15, 9))

    for i in range(number_of_channels):

        signal = data[i] * 1e6

        plt.plot(
            times,
            signal + i * 150,
            linewidth=0.6,
            label=raw.ch_names[i]
        )

    plt.xlim(0, min(10, times[-1]))

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (µV, offset)")
    plt.title("NEUROSYNC - EEG Waveforms")

    plt.legend(
        loc="upper right",
        fontsize=8
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "physionet_eeg_waveforms.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # 6. POWER SPECTRAL DENSITY
    # ==========================================

    print("Generating EEG frequency spectrum...")

    spectrum = raw.compute_psd(
        method="welch",
        fmin=1,
        fmax=40,
        verbose=False
    )

    psd, freqs = spectrum.get_data(
        return_freqs=True
    )

    mean_psd = np.mean(psd, axis=0)

    plt.figure(figsize=(12, 6))

    plt.plot(
        freqs,
        mean_psd,
        linewidth=2
    )

    plt.axvspan(
        1, 4,
        alpha=0.15,
        label="Delta"
    )

    plt.axvspan(
        4, 8,
        alpha=0.15,
        label="Theta"
    )

    plt.axvspan(
        8, 13,
        alpha=0.15,
        label="Alpha / Mu"
    )

    plt.axvspan(
        13, 30,
        alpha=0.15,
        label="Beta"
    )

    plt.axvspan(
        30, 40,
        alpha=0.15,
        label="Gamma"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("NEUROSYNC - EEG Frequency Spectrum")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "physionet_eeg_psd.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # 7. EXTRACT EEG EVENTS
    # ==========================================

    print("\nExtracting EEG events...")

    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print("\nAvailable event labels:")

    for name, code in event_id.items():
        print(name, ":", code)

    # ==========================================
    # 8. SELECT MOTOR IMAGERY EVENTS
    # ==========================================

    selected_events = {
        key: value
        for key, value in event_id.items()
        if key in ["T1", "T2"]
    }

    if len(selected_events) != 2:
        raise ValueError(
            "Expected both T1 and T2 events. "
            "Check the recording annotations."
        )

    # ==========================================
    # 9. CREATE EEG EPOCHS
    # ==========================================

    print("\nCreating EEG epochs...")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_events,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )

    epoch_data = epochs.get_data()

    # ==========================================
    # 10. EXTRACT CLASSIFICATION LABELS
    # ==========================================

    labels = epochs.events[:, -1]

    print("\nCLASS DISTRIBUTION")
    print("------------------")

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples")

    # ==========================================
    # 11. SAVE PROCESSED EEG DATASET
    # ==========================================

    np.savez_compressed(
        processed_dir / "eeg_dataset.npz",
        X=epoch_data,
        y=labels,
        channels=np.array(epochs.ch_names),
        sampling_rate=raw.info["sfreq"]
    )

    print("\nProcessed dataset saved successfully!")

    # ==========================================
    # 12. SAVE EPOCH SUMMARY
    # ==========================================

    summary = f"""
NEUROSYNC - EEG EPOCH INFORMATION
=================================

Number of epochs: {len(epochs)}

Epoch array shape: {epoch_data.shape}

Number of channels: {len(epochs.ch_names)}

Sampling frequency: {raw.info["sfreq"]} Hz

Event mapping: {selected_events}

Class distribution:
{dict(zip(unique.tolist(), counts.tolist()))}
"""

    with open(
        output_dir / "physionet_epoch_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(summary)

    # ==========================================
    # 13. FINAL OUTPUT
    # ==========================================

    print("\nEEG EPOCH INFORMATION")
    print("---------------------")

    print("Number of epochs:", len(epochs))
    print("Epoch array shape:", epoch_data.shape)
    print("Event mapping:", selected_events)

    print("\nGenerated files:")
    print("1. outputs/physionet_eeg_waveforms.png")
    print("2. outputs/physionet_eeg_psd.png")
    print("3. outputs/physionet_epoch_summary.txt")
    print("4. data/processed/eeg_dataset.npz")

    print("\nNEUROSYNC EEG exploration completed!")


if __name__ == "__main__":
    run_physionet()
