# SpectralFace: Cross-Spectral Face Recognition System

SpectralFace is an advanced biometric system designed to perform face recognition across different spectral domains—specifically between **Visible (VIS)** and **Near-Infrared (NIR)** light. This technology is critical for surveillance, nighttime security, and robust identity verification where lighting conditions vary.

![UI Preview](https://img.shields.io/badge/Status-Beta-blue)

## 🚀 Key Features

- **Cross-Spectral Comparison**: Match a Near-Infrared (NIR) probe image against a Visible (VIS) reference.
- **Identity Recognition**: Search a live or uploaded NIR probe against a pre-enrolled VIS gallery.
- **Explainable AI (XAI)**: Visualizes the neural network's focus areas using activation heatmaps.
- **Real-time Processing**: Live face detection and recognition via webcam.
- **Analytics Dashboard**: Comprehensive system stats and demographic distributions.

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask, PyTorch
- **Computer Vision**: OpenCV (Haar Cascades), PIL
- **Machine Learning**: ResNet-18 backbone with Triplet and Softmax losses.
- **Frontend**: HTML5, Vanilla CSS3 (Modern Glassmorphism Design), JavaScript (ES6).

## 📥 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AymenSensai/master-project.git
cd master-project
```

### 2. Prepare the Environment

Initialize the required directory structure:

```bash
python3 setup_project.py
```

### 3. Create a Virtual Environment (Recommended)

To avoid "externally managed environment" errors on macOS/Linux, it is highly recommended to use a virtual environment:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 4. Install Dependencies

Once the virtual environment is active, install the required libraries:

```bash
pip install -r requirements.txt
```

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

To evaluate a trained checkpoint:

```bash
python3 main.py eval --config config.yaml --checkpoint checkpoints/model_best.pth
```

## 🧠 System Architecture & Logic

### 1. Model Architecture

The system utilizes a **ResNet-18** deep learning backbone. It transforms raw face images into a **512-dimensional embedding** (a mathematical representation of identity).

### 2. Training Strategy (VIS-NIR Alignment)

To bridge the "spectral gap" (differences between visible and infrared light), we use a **Dual-Loss approach**:

- **Softmax Loss (Identity Classification)**: Forces the model to learn features that distinguish between different individuals.
- **Triplet Loss (Spectral Invariance)**: Given an NIR "Anchor," it pulls a VIS "Positive" (same person) closer while pushing a "Negative" (different person) further away in the embedding space. This teaches the model to ignore light spectra and focus solely on identity.

### 3. Evaluation Protocol

The system uses a **Gallery-Probe** retrieval model:

- **Gallery**: A database of VIS (Visible) images for all known identities.
- **Probe**: An "unknown" NIR image captured from a sensor.
- **Metric**: We calculate **Cosine Similarity** between the probe and every gallery entry. The system reports **Rank-1 Accuracy** (is the top match correct?) and **VR@FAR** (Verification Rate at a specific False Acceptance Rate).

### 4. Explainable AI (XAI)

The application implements **Grad-CAM** (Gradient-weighted Class Activation Mapping). It visualizes the pixels (eyes, nose, jawline) that most influenced the model's decision, providing transparency into the recognition process.

## 📂 Project Structure

- `app.py`: Main Flask application entry point.
- `model.py`: Architecture definition (features and classification layers).
- `gallery_manager.py`: Logic for managing the identity gallery and similarity search.
- `dataset.py`: Custom PyTorch dataset loader for TuftsFaceDatabase.
- `xai_utils.py`: Utility for generating Explainable AI heatmaps.
- `gallery/`: Identity storage folder (structure: `gallery/Identity_Name/*.jpg`).
- `static/` & `templates/`: Frontend assets and HTML.

## 🔍 How it Works

1. **Preprocessing**: Images are resized to 112x112 and normalized.
2. **Feature Extraction**: A ResNet-18 backbone extracts a 512-dimensional embedding.
3. **Domain Alignment**: The network is trained using a combination of Cross-Entropy and Triplet Loss to bridge the gap between VIS and NIR sensors.
4. **Matching**: Similarity is measured using Cosine Distance.
5. **Visualization**: Grad-CAM heatmaps highlight pixels contributing most to the identity decision.