# 🧠 NEUROSYNC — Neural Signal Intelligence

### EEG-Based Motor Imagery Classification Using Deep Learning

NEUROSYNC is a research-oriented EEG signal processing and deep learning project that investigates the classification of left-hand and right-hand motor imagery using electroencephalography (EEG) signals.

The project combines EEG preprocessing, conventional machine learning, deep learning architectures, subject-independent evaluation, architectural ablation studies, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Motor imagery refers to the mental rehearsal of a movement without physically executing it. EEG signals can capture changes in brain activity associated with motor imagery tasks.

NEUROSYNC explores how deep learning models can learn patterns from EEG signals and classify two motor imagery categories:

- Left-hand motor imagery
- Right-hand motor imagery

The project investigates EEGNet as the primary classification model and introduces NeuroFusion-T, a multi-branch research architecture designed to explore spatial, temporal, and frequency-domain EEG representations.

## 🎯 Objectives

- Process and analyze EEG motor imagery signals.
- Develop machine learning and deep learning classification models.
- Investigate spatial, temporal, and frequency-domain representations.
- Evaluate generalization to unseen subjects using Leave-One-Subject-Out (LOSO) validation.
- Conduct architectural ablation experiments.
- Develop an interactive EEG visualization and prediction dashboard.

---

## 📂 Dataset

**Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

The project uses recordings from 10 subjects and selected motor imagery runs.

| Dataset property | Description |
|---|---|
| Subjects | 10 |
| EEG channels | 64 |
| Selected runs | 4, 8, 12 |
| Sampling frequency | 160 Hz |
| Epoch duration | 4 seconds |
| Samples per epoch | 641 |
| Total epochs | 450 |
| Classification classes | 2 |

### Class mapping

| Original event label | Project class |
|---|---|
| T1 (event 2) | Left-hand motor imagery (0) |
| T2 (event 3) | Right-hand motor imagery (1) |

The dataset is used for research experimentation and model development.

Dataset source:

[PhysioNet — EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/)

---

## ⚙️ EEG Preprocessing

The EEG preprocessing pipeline includes:

1. Loading EEG recordings from the selected subjects and runs.
2. Extracting labelled motor imagery epochs.
3. Filtering signals between 1 Hz and 40 Hz.
4. Organizing EEG epochs into a consistent channel-by-time representation.
5. Mapping original event labels into two classification classes.
6. Saving the processed multisubject dataset.

The resulting dataset has the shape:

`(450, 64, 641)`

For model evaluation, channel-wise normalization statistics are calculated using only the training subjects in each fold to avoid test-subject information leakage.

---

## 🧠 Model Architectures

### 1. EEGNet — Primary Classification Model

EEGNet is a compact convolutional neural network designed for EEG-based brain-computer interface applications.

The implementation includes:

- Temporal convolution
- Depthwise spatial convolution
- Separable convolution
- Batch normalization
- ELU activation
- Average pooling
- Dropout regularization
- Fully connected classification layer

EEGNet was selected as the primary model because it achieved the strongest performance among the evaluated deep learning architectures in the controlled LOSO experiments.

### 2. NeuroFusion-T — Research Architecture

NeuroFusion-T is a custom multi-branch architecture developed to investigate complementary EEG representations.

It includes three feature extraction branches:

**Spatial branch**
- Learns spatial relationships across EEG channels.

**Temporal branch**
- Extracts temporal patterns from EEG signals.

**Frequency branch**
- Uses differentiable FFT-based features.
- Investigates Mu/Alpha (8–13 Hz) and Beta (13–30 Hz) frequency bands.

**Attention-based fusion**
- Combines branch representations using multi-head attention.
- Applies residual connections and layer normalization.
- Passes fused representations to the classification layer.

NeuroFusion-T is retained as an experimental research architecture. It did not outperform EEGNet in the current experiments.

---

## 🔬 Experimental Methodology

### Leave-One-Subject-Out (LOSO) Validation

The project uses Leave-One-Subject-Out cross-validation to evaluate generalization to unseen subjects.

The procedure:

1. Hold out one subject for testing.
2. Use the remaining nine subjects for training and model selection.
3. Split the training subjects into inner training and validation groups.
4. Select training duration using validation loss and early stopping.
5. Retrain using all nine outer training subjects.
6. Evaluate on the held-out subject.
7. Repeat until every subject has been used as the test subject.

EEGNet was evaluated using three random seeds:

- 11
- 22
- 33

The controlled validation experiment contains 30 selected epoch counts. The median selected training duration was 8 epochs.

### Evaluation Metrics

- Accuracy
- Balanced accuracy
- Confusion matrix
- Mean and standard deviation across seeds
- Majority-vote ensemble performance

---

## 📊 Controlled Validation Results

### EEGNet vs CSP + LDA

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| CSP + LDA | 49.78% | 49.43% |
| EEGNet — Seed 11 | 78.89% | 78.82% |
| EEGNet — Seed 22 | 74.44% | 74.39% |
| EEGNet — Seed 33 | 79.78% | 79.70% |
| EEGNet — Mean | **77.70%** | **77.64%** |
| EEGNet — Majority Vote | **78.89%** | **78.83%** |

EEGNet demonstrated stronger subject-independent classification performance than the CSP + LDA baseline in this controlled experiment.

### EEGNet Mean Performance

- Mean accuracy: 77.70%
- Accuracy standard deviation: 2.86%
- Mean balanced accuracy: 77.64%
- Balanced accuracy standard deviation: 2.85%

These results represent the controlled LOSO evaluation, not the performance of the final all-subject demonstration model.

---

## 🧪 NeuroFusion-T Ablation Study

An ablation study was conducted to investigate the contribution of different feature extraction branches.

Seven configurations were evaluated:

1. Spatial only
2. Temporal only
3. Frequency only
4. Spatial + Temporal
5. Spatial + Frequency
6. Temporal + Frequency
7. Spatial + Temporal + Frequency

### Ablation Results

| Configuration | Mean Accuracy | Balanced Accuracy |
|---|---:|---:|
| Spatial | 59.26% | 59.11% |
| Temporal | 69.48% | 69.50% |
| Frequency | 49.56% | 49.36% |
| Spatial + Temporal | **69.85%** | 69.74% |
| Spatial + Frequency | 53.93% | 53.91% |
| Temporal + Frequency | 68.89% | 68.80% |
| Full NeuroFusion-T | 68.89% | 68.83% |

### Key Observations

- The spatial + temporal configuration achieved the highest mean accuracy among the tested NeuroFusion-T configurations.
- The temporal branch alone demonstrated relatively stable performance.
- The frequency-only configuration performed close to chance level.
- Adding the frequency branch did not improve the tested combinations.
- The full NeuroFusion-T model did not outperform the spatial + temporal configuration.
- EEGNet remained the strongest evaluated deep learning model.

The ablation results help identify which feature representations contributed most effectively in the current experimental setting.

---

## 💻 NEUROSYNC Interactive Dashboard

The project includes an interactive Streamlit dashboard for visualizing prerecorded EEG signals and model predictions.

### Dashboard Features

**Signal Configuration**
- EEG channel selection
- Sampling frequency display
- Dataset information
- Model information

**Signal Monitoring**
- EEG waveform visualization
- Frequency spectrum visualization
- Simulated signal streaming
- Start, Stop, and Reset controls

**AI Prediction**
- Left-hand motor imagery prediction
- Right-hand motor imagery prediction
- Prediction confidence
- Ensemble probability visualization

**Prediction History**
- Epoch number
- Predicted class
- Confidence
- Processing timestamp

**Research Performance**
- Controlled LOSO accuracy
- Balanced accuracy
- Majority-vote accuracy

The dashboard uses an ensemble of three trained EEGNet models with seeds 11, 22, and 33.

The current dashboard uses prerecorded EEG epochs for simulated streaming. It is not connected to live EEG acquisition hardware.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computation |
| MNE-Python | EEG processing |
| SciPy | Signal processing |
| Scikit-learn | Machine learning and evaluation |
| PyTorch | Deep learning |
| Matplotlib | Signal visualization |
| Plotly | Interactive visualizations |
| Streamlit | Interactive dashboard |
| Git and GitHub | Version control |

---

## 📁 Project Structure

```text
NEUROSYNC/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── data/
│   └── processed/
│       └── multisubject_eeg.npz
│
├── outputs/
│   ├── controlled_validation_results.txt
│   ├── neurofusion_ablation_results.txt
│   └── neurosync_eegnet_ensemble.pt
│
├── src/
│   ├── physionet.py
│   ├── multisubject.py
│   ├── baseline.py
│   ├── loso.py
│   ├── frequency_loso.py
│   ├── model_benchmark.py
│   ├── eegnet_loso.py
│   ├── validate_eegnet.py
│   ├── neurofusion_t.py
│   ├── neurofusion_ablation.py
│   └── train_final_eegnet.py
│
└── .venv/
```

Generated checkpoints, reports, and processed datasets may be stored locally or managed through Git LFS, depending on repository configuration.

---

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd NEUROSYNC
```

Replace the placeholder with the actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Run the multisubject preprocessing script:

```bash
python src/multisubject.py
```

This script prepares the multisubject processed EEG dataset.

### 5. Train the final EEGNet ensemble

```bash
python src/train_final_eegnet.py
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard loads the trained model package and processed EEG dataset from their expected project paths.

---

## ⚠️ Limitations

- The current dataset contains 10 subjects and 450 selected EEG epochs.
- The results are preliminary and require validation on larger and more diverse datasets.
- The dashboard uses prerecorded EEG signals rather than real-time hardware acquisition.
- The final demonstration ensemble was trained using all available samples; its dashboard predictions are not independent test predictions.
- Model confidence represents predicted class probability, not a calibrated measure of clinical certainty.
- The project is an experimental motor imagery classification prototype, not a medical diagnostic system.

---

## 🔮 Future Enhancements

- Integration with live EEG acquisition hardware.
- Real-time EEG preprocessing and signal streaming.
- Evaluation on larger multisubject EEG datasets.
- Additional cross-dataset validation.
- Improved model generalization.
- Further investigation of spatial-temporal-attention architectures.
- Model explainability and feature visualization.
- Deployment of the dashboard as an accessible web application.

---

## 👩‍💻 Project Focus

NEUROSYNC explores the intersection of:

- Brain-computer interfaces
- EEG signal processing
- Deep learning
- Attention-based neural networks
- Motor imagery classification
- Human-computer interaction

The project aims to investigate how machine learning can extract useful patterns from EEG signals for experimental brain-computer interface applications.

---

**NEUROSYNC — Neural Signal Intelligence**

*Exploring EEG patterns through signal processing, deep learning, and intelligent classification.*
