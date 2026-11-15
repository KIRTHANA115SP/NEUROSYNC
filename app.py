
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

from pathlib import Path
from scipy.signal import welch
from datetime import datetime


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NEUROSYNC | Neural Signal Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "multisubject_eeg.npz"
MODEL_PATH = ROOT / "outputs" / "neurosync_eegnet_ensemble.pt"

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


# =====================================================
# 2. PROFESSIONAL DARK UI
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080e1a, #101b2e);
    color: #e8f1ff;
}

section[data-testid="stSidebar"] {
    background: #101a2b;
    border-right: 1px solid #263b55;
}

h1 {
    color: #72ddff !important;
    font-weight: 800 !important;
    letter-spacing: 2px;
}

h2, h3 {
    color: #72ddff !important;
}

p, label {
    color: #d4e1f2;
}

div[data-testid="stMetric"] {
    background: #17263b;
    border: 1px solid #29435f;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #72ddff;
}

div[data-testid="stMetricLabel"] {
    color: #a9c2df;
}

.stButton button {
    border-radius: 10px;
    border: 1px solid #315574;
    background: #172a43;
    color: #72ddff;
    font-weight: 600;
    min-height: 45px;
}

.stButton button:hover {
    background: #244565;
    color: white;
    border-color: #72ddff;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #36c8ff;
}

hr {
    border-color: #29405c;
}

.neuro-header {
    background: linear-gradient(110deg, #14263e, #102d43);
    border: 1px solid #315574;
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
}

.neuro-subtitle {
    color: #a8c6df;
    font-size: 15px;
}

.neuro-card {
    background: #14243a;
    border: 1px solid #29435f;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.status-live {
    color: #48e6a1;
    font-weight: bold;
}

.status-paused {
    color: #ffc66d;
    font-weight: bold;
}

.small-note {
    color: #91a8c3;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. EEGNET ARCHITECTURE
# =====================================================

class EEGNet(nn.Module):

    def __init__(self, n_channels, n_times, n_classes=2):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1, 8, (1, 64),
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(8)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                8, 16,
                (n_channels, 1),
                groups=8,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                16, 16,
                (1, 16),
                padding=(0, 8),
                groups=16,
                bias=False
            ),
            nn.Conv2d(
                16, 16,
                (1, 1),
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1, 1, n_channels, n_times
            )

            size = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(size, n_classes)

    def features(self, x):

        x = self.temporal(x)
        x = self.spatial(x)

        return self.separable(x)

    def forward(self, x):

        x = self.features(x).flatten(1)

        return self.classifier(x)


# =====================================================
# 4. LOAD DATASET AND TRAINED MODELS
# =====================================================

@st.cache_resource
def load_resources():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    data = np.load(DATA_PATH)

    X = data["X"].astype(np.float32)

    y = np.where(
        data["y"] == 2, 0, 1
    ).astype(np.int64)

    groups = (
        data["groups"]
        if "groups" in data.files
        else np.zeros(len(X))
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    models = []

    for seed in checkpoint["seeds"]:

        model = EEGNet(
            checkpoint["n_channels"],
            checkpoint["n_times"],
            checkpoint["n_classes"]
        ).to(DEVICE)

        model.load_state_dict(
            checkpoint["model_states"][seed]
        )

        model.eval()

        models.append(model)

    mean = checkpoint["channel_mean"].cpu().numpy()
    std = checkpoint["channel_std"].cpu().numpy()

    channels = checkpoint["channels"]

    if not channels:
        channels = [
            f"Channel {i + 1}"
            for i in range(X.shape[1])
        ]

    sampling_rate = checkpoint["sampling_rate"]

    if sampling_rate is None:
        sampling_rate = 160.0

    return (
        X, y, groups, models,
        mean, std, channels,
        sampling_rate, checkpoint
    )


# =====================================================
# 5. ENSEMBLE PREDICTION
# =====================================================

def predict_epoch(epoch, models, mean, std):

    normalized = (epoch - mean[0]) / std[0]

    tensor = torch.tensor(
        normalized[np.newaxis, np.newaxis, :, :],
        dtype=torch.float32
    )

    probabilities = []

    with torch.no_grad():

        for model in models:

            logits = model(tensor)

            probs = torch.softmax(
                logits, dim=1
            )

            probabilities.append(
                probs.cpu().numpy()[0]
            )

    ensemble_probs = np.mean(
        probabilities,
        axis=0
    )

    predicted_class = int(
        np.argmax(ensemble_probs)
    )

    confidence = float(
        ensemble_probs[predicted_class] * 100
    )

    return predicted_class, confidence, ensemble_probs


# =====================================================
# 6. SESSION STATE
# =====================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "epoch_index" not in st.session_state:
    st.session_state.epoch_index = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0


# =====================================================
# 7. LOAD APPLICATION
# =====================================================

try:

    (
        X, y, groups, models,
        mean, std, channels,
        sampling_rate, checkpoint
    ) = load_resources()

except Exception as error:

    st.error(f"NEUROSYNC loading error: {error}")

    st.info(
        "Verify that the dataset and trained model "
        "exist in their expected project folders."
    )

    st.stop()


# =====================================================
# 8. HEADER
# =====================================================

st.markdown("""
<div class="neuro-header">

<h1>🧠 NEUROSYNC</h1>

<div class="neuro-subtitle">
Neural Signal Intelligence
<br>
EEG Motor Imagery Classification System
</div>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 9. SIDEBAR
# =====================================================

st.sidebar.markdown("## ⚙️ Signal Configuration")

selected_channel = st.sidebar.selectbox(
    "Select EEG Channel",
    channels
)

channel_index = channels.index(selected_channel)

st.sidebar.divider()

st.sidebar.markdown("### Dataset Information")

st.sidebar.write(f"**Sampling Rate:** {sampling_rate:g} Hz")
st.sidebar.write(f"**EEG Channels:** {len(channels)}")
st.sidebar.write(f"**Available Epochs:** {len(X)}")
st.sidebar.write(f"**Subjects:** {len(np.unique(groups))}")

st.sidebar.divider()

st.sidebar.markdown("### Model Information")

st.sidebar.write("**Architecture:** EEGNet")
st.sidebar.write("**Ensemble Models:** 3")
st.sidebar.write("**Random Seeds:** 11, 22, 33")
st.sidebar.write("**Training Epochs:** 8")

st.sidebar.divider()

st.sidebar.markdown("""
<div class="small-note">
NEUROSYNC is an experimental research prototype.
It uses prerecorded EEG signals for simulated streaming.
It is not a medical diagnostic system.
</div>
""", unsafe_allow_html=True)


# =====================================================
# 10. STREAMING CONTROLS
# =====================================================

st.markdown("## 🎮 Signal Controls")

start_col, stop_col, reset_col = st.columns(3)

with start_col:

    if st.button(
        "▶ START STREAM",
        use_container_width=True
    ):

        st.session_state.running = True

with stop_col:

    if st.button(
        "⏸ STOP STREAM",
        use_container_width=True
    ):

        st.session_state.running = False

with reset_col:

    if st.button(
        "↻ RESET",
        use_container_width=True
    ):

        st.session_state.running = False
        st.session_state.epoch_index = 0
        st.session_state.history = []
        st.session_state.total_processed = 0


st.divider()


# =====================================================
# 11. MAIN DASHBOARD
# =====================================================

@st.fragment(run_every="1s")
def dashboard_stream():

    if st.session_state.running:

        epoch_index = st.session_state.epoch_index

    else:

        epoch_index = st.session_state.epoch_index

    epoch = X[epoch_index]

    predicted_class, confidence, probs = predict_epoch(
        epoch,
        models,
        mean,
        std
    )

    label = (
        "LEFT-HAND MOTOR IMAGERY"
        if predicted_class == 0
        else "RIGHT-HAND MOTOR IMAGERY"
    )

    # ---------------------------------------------
    # STATUS
    # ---------------------------------------------

    if st.session_state.running:

        st.markdown(
            '<span class="status-live">● STREAMING</span>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<span class="status-paused">● PAUSED</span>',
            unsafe_allow_html=True
        )

    # ---------------------------------------------
    # PREDICTION PANEL
    # ---------------------------------------------

    st.markdown("## 🧠 AI Prediction")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Current EEG Epoch",
            f"{epoch_index + 1} / {len(X)}"
        )

    with c2:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    if predicted_class == 0:

        st.success(
            "✋ LEFT-HAND MOTOR IMAGERY"
        )

    else:

        st.info(
            "✋ RIGHT-HAND MOTOR IMAGERY"
        )

    st.progress(confidence / 100)

    # ---------------------------------------------
    # PROBABILITIES
    # ---------------------------------------------

    st.markdown("### Model Prediction Probabilities")

    left_col, right_col = st.columns(2)

    with left_col:

        st.metric(
            "Left-hand probability",
            f"{probs[0] * 100:.2f}%"
        )

    with right_col:

        st.metric(
            "Right-hand probability",
            f"{probs[1] * 100:.2f}%"
        )

    st.divider()

    # ---------------------------------------------
    # EEG WAVEFORM
    # ---------------------------------------------

    st.markdown("## 📈 EEG Signal Waveform")

    signal = epoch[channel_index]

    time_axis = (
        np.arange(len(signal)) / sampling_rate
    )

    scale = (
        1e6
        if np.max(np.abs(signal)) < 1e-2
        else 1
    )

    signal_display = signal * scale

    waveform = go.Figure()

    waveform.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal_display,
            mode="lines",
            name=selected_channel,
            line=dict(
                color="#55c7ff",
                width=1.8
            ),
            fill="tozeroy",
            fillcolor="rgba(85,199,255,0.07)"
        )
    )

    waveform.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,27,45,0.8)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (µV)",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )

    st.plotly_chart(
        waveform,
        use_container_width=True
    )

    # ---------------------------------------------
    # FREQUENCY SPECTRUM
    # ---------------------------------------------

    st.markdown("## 📊 Frequency Spectrum")

    frequencies, power = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(256, len(signal))
    )

    spectrum = go.Figure()

    spectrum.add_trace(
        go.Scatter(
            x=frequencies,
            y=10 * np.log10(power + 1e-20),
            mode="lines",
            name="Power Spectrum",
            line=dict(
                color="#a78bfa",
                width=2
            ),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)"
        )
    )

    spectrum.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,27,45,0.8)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB/Hz)",
        height=320,
        xaxis=dict(range=[1, 40]),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )

    st.plotly_chart(
        spectrum,
        use_container_width=True
    )

    # ---------------------------------------------
    # STREAM HISTORY
    # ---------------------------------------------

    if st.session_state.running:

        st.session_state.history.append({
            "Epoch": epoch_index + 1,
            "Prediction": label,
            "Confidence": round(confidence, 2),
            "Time": datetime.now().strftime("%H:%M:%S")
        })

        st.session_state.total_processed += 1

        st.session_state.epoch_index = (
            epoch_index + 1
        ) % len(X)

    st.divider()

    st.markdown("## 📋 Prediction History")

    if st.session_state.history:

        history = st.session_state.history[-20:]

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Press START STREAM to begin simulated EEG streaming."
        )

    st.caption(
        f"Epochs processed during this session: "
        f"{st.session_state.total_processed}"
    )


dashboard_stream()


# =====================================================
# 12. RESEARCH PERFORMANCE
# =====================================================

st.divider()

st.markdown("## 🔬 Research Performance")

st.caption(
    "Controlled Leave-One-Subject-Out (LOSO) validation results"
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "EEGNet Accuracy",
    "77.70%"
)

m2.metric(
    "Balanced Accuracy",
    "77.64%"
)

m3.metric(
    "Majority Vote Accuracy",
    "78.89%"
)

st.info(
    "These results come from controlled offline LOSO evaluation. "
    "They are not live dashboard accuracy measurements."
)


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:20px;">

<h3>🧠 NEUROSYNC</h3>

<p style="color:#91a8c3;">
Neural Signal Intelligence
</p>

<p style="color:#71869f; font-size:12px;">
EEG Motor Imagery Classification Research Prototype
<br>
Powered by EEGNet | PyTorch | Streamlit
<br><br>
Experimental research software — not for medical diagnosis.
</p>

</div>
""", unsafe_allow_html=True)
