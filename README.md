# SpectralFace: Cross-Spectral Face Recognition System

SpectralFace is an advanced biometric system designed to perform face recognition across different spectral domains—specifically between **Visible (VIS)** and **Near-Infrared (NIR)** light. This technology is critical for surveillance, nighttime security, and robust identity verification where lighting conditions vary.

![UI Preview](https://img.shields.io/badge/Status-Beta-blue)

## 🚀 Key Features

- **Cross-Spectral Verification**: Match a Near-Infrared (NIR) probe image against a Visible (VIS) reference using cosine similarity.
- **Identity Recognition**: Search a live or uploaded NIR probe against a pre-enrolled VIS gallery.
- **Live Webcam Recognition**: Real-time face detection and recognition with temporal score smoothing and a cyberpunk HUD overlay.
- **Gallery Browser**: Browse all enrolled VIS identities stored in the dataset.
- **Analytics Dashboard**: Displays identities count, VIS/NIR image counts, and compute device in use.
- **NIR-VIS Challenge Mode**: An interactive game where the user competes against the AI to determine whether two cross-spectral faces belong to the same person.
- **Explainable AI (XAI)**: Visualizes the neural network's focus areas using Grad-CAM activation heatmaps.

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask, PyTorch
- **Computer Vision**: OpenCV (Haar Cascades), PIL
- **Machine Learning**: Two-Stream ResNet-18 with shared convolutional weights, trained with a combined Cross-Entropy and Triplet Loss.
- **Frontend**: HTML5, Vanilla CSS3 (Modern Glassmorphism Design), JavaScript (ES6).

## 📥 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AymenSensai/master-project.git
cd master-project
```

### 2. Create a Virtual Environment (Recommended)

To avoid "externally managed environment" errors on macOS/Linux, it is highly recommended to use a virtual environment:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 3. Install Dependencies

Once the virtual environment is active, install the required libraries:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the Tufts Face Database under `data/TuftsFaceDatabase/` with the following structure:

```
data/TuftsFaceDatabase/
├── VIS/
│   ├── Identity_Name/
│   │   └── *.jpg
│   └── ...
└── NIR/
    ├── Identity_Name/
    │   └── *.jpg
    └── ...
```

The VIS folder is used as the recognition gallery. The NIR folder is used for challenge mode probes.

## 🏃 Running the Application

### Start the Web Server

Launch the Flask application:

```bash
python3 app.py
```

Open your browser and navigate to: **`http://127.0.0.1:5001`**

### Training & Evaluation (Optional)

If you have the Tufts Face Database dataset, you can train the model using:

```bash
python3 main.py train --config config.yaml
```

To resume training from the best checkpoint:

```bash
python3 main.py train --config config.yaml --resume
```

To evaluate a trained checkpoint:

```bash
    python3 main.py eval --config config.yaml --checkpoint checkpoints/model_best.pth
```

## 🧠 System Architecture & Logic

### 1. Model Architecture

The system uses a **Two-Stream ResNet-18** (`TwoStreamResNet`) with **shared convolutional weights** across both the VIS and NIR domains. The shared backbone extracts features, which are then projected into a **512-dimensional embedding** space and **L2-normalized** onto a unit hypersphere for stable cosine similarity comparisons.

### 2. Training Strategy (VIS-NIR Alignment)

To bridge the "spectral gap" (differences between visible and infrared light), the model is trained with a **combined `CrossSpectralLoss`**:

- **Cross-Entropy Loss (Softmax)**: Forces the model to learn discriminative features that distinguish between different individuals.
- **Triplet Loss (Spectral Invariance)**: Given an NIR "Anchor," it pulls a VIS "Positive" (same person) closer in embedding space while pushing a "Negative" (different person) further away. This teaches the model to ignore light spectra and focus solely on identity. Uses online hard negative mining within each batch.

### 3. Evaluation Protocol

The system uses a **Gallery-Probe** retrieval model:

- **Gallery**: A database of VIS (Visible) images for all known identities, stored under `data/TuftsFaceDatabase/VIS/`.
- **Probe**: An "unknown" NIR image captured from a sensor.
- **Metric**: Cosine Similarity between the probe embedding and every gallery entry. The system reports **Rank-1 Accuracy** and **VR@FAR** (Verification Rate at a specific False Acceptance Rate).

### 4. Explainable AI (XAI)

The application implements **Grad-CAM** (Gradient-weighted Class Activation Mapping). It visualizes the pixels (eyes, nose, jawline) that most influenced the model's decision, providing transparency into the recognition process.

## 📂 Project Structure

- `app.py`: Main Flask application entry point and API routes.
- `model.py`: `TwoStreamResNet` architecture definition (shared backbone + embedding + optional classifier).
- `losses.py`: `TripletLoss` and `CrossSpectralLoss` (combined CE + Triplet) definitions.
- `train.py`: Training loop logic.
- `evaluate.py`: Evaluation logic (Rank-1, VR@FAR).
- `main.py`: CLI entry point for `train` and `eval` subcommands.
- `gallery_manager.py`: Logic for managing the identity gallery and cosine similarity search.
- `dataset.py`: Custom PyTorch dataset loader for TuftsFaceDatabase.
- `xai_utils.py`: Utility for generating Grad-CAM Explainable AI heatmaps.
- `utils.py`: Shared utilities (config loading, face detection & cropping).
- `config.yaml`: All hyperparameters (dataset paths, training, evaluation).
- `data/TuftsFaceDatabase/VIS/`: VIS identity gallery (structure: `VIS/Identity_Name/*.jpg`).
- `data/TuftsFaceDatabase/NIR/`: NIR probe images for challenge mode (structure: `NIR/Identity_Name/*.jpg`).
- `checkpoints/`: Saved model checkpoints (`model_best.pth`).
- `static/` & `templates/`: Frontend assets and HTML.

## 🔍 How it Works

1. **Preprocessing**: Images are resized to 112×112 and normalized to `[-1, 1]`.
2. **Face Detection**: OpenCV Haar Cascade detects and crops the face region with padding before embedding.
3. **Feature Extraction**: The shared Two-Stream ResNet-18 backbone extracts features, projected to a 512-dim L2-normalized embedding.
4. **Domain Alignment**: The network is trained using Cross-Entropy and Triplet Loss to bridge the gap between VIS and NIR sensors.
5. **Matching**: Similarity is measured using Cosine Distance (dot product of L2-normalized embeddings).
6. **Visualization**: Grad-CAM heatmaps highlight pixels contributing most to the identity decision.
