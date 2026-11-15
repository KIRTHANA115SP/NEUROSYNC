from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def run_demo():
    """Generate a synthetic EEG-like signal to verify the local pipeline."""
    print("\nNEUROSYNC synthetic EEG smoke test")
    print("----------------------------------")
    print("NOTE: These are simulated signals, not real EEG recordings.")

    rng = np.random.default_rng(42)
    sfreq = 128
    duration = 8
    times = np.arange(0, duration, 1 / sfreq)
    channel_names = ["C3-like", "Cz-like", "C4-like"]

    signals = []
    for idx in range(3):
        alpha = 12 + idx
        beta = 18 + idx
        signal = (
            18e-6 * np.sin(2 * np.pi * alpha * times)
            + 7e-6 * np.sin(2 * np.pi * beta * times)
            + 5e-6 * rng.standard_normal(times.size)
        )
        signals.append(signal)

    data = np.asarray(signals)
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(channel_names), 1, figsize=(12, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(times, data[i] * 1e6, linewidth=0.7)
        ax.set_ylabel(f"{channel_names[i]}\nµV")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("NEUROSYNC — Synthetic EEG-like waveforms (simulation only)")
    fig.tight_layout()
    fig.savefig(out / "demo_eeg_waveforms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(channel_names):
        freqs, psd = welch(data[i], fs=sfreq, nperseg=sfreq * 2)
        ax.semilogy(freqs, psd, label=name)
    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (V²/Hz)")
    ax.set_title("Synthetic signal power spectrum")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "demo_eeg_psd.png", dpi=150)
    plt.close(fig)

    print("Created:")
    print(" - outputs/demo_eeg_waveforms.png")
    print(" - outputs/demo_eeg_psd.png")
    print("\nSmoke test complete. Next, try: python main.py --physionet")
