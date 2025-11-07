# � Speech Emotion Recognition (SER)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning-based Speech Emotion Recognition system using **CNN + BiLSTM with Attention** architecture. This project classifies audio speech samples into 6 emotion categories with high accuracy using advanced feature extraction and neural network techniques.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Datasets](#-datasets)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Training Details](#-training-details)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This Speech Emotion Recognition system uses state-of-the-art deep learning techniques to identify emotions from audio speech. The model can classify speech into six emotion categories:

- 😐 **Neutral**
- 😊 **Happy**
- 😢 **Sad**
- 😠 **Angry**
- 😨 **Fearful**
- 🤢 **Disgust**

### Key Highlights

- **Advanced Feature Engineering**: MFCC + Delta + Delta-Delta features with normalization
- **Hybrid Architecture**: CNN for spatial feature extraction + BiLSTM for temporal modeling
- **Attention Mechanism**: Weighted temporal pooling for better context understanding
- **GPU Optimized**: Fully optimized for NVIDIA RTX 3060 Ti (8GB VRAM)
- **Smart Caching**: Pre-computed MFCC features for faster training iterations
- **Production-Ready**: Includes early stopping, learning rate scheduling, and gradient clipping

---

## ✨ Features

### 🔊 Audio Processing
- **Sample Rate**: 16kHz (standardized)
- **Feature Extraction**: 40 MFCC coefficients + deltas + delta-deltas (120 features total)
- **GPU-Accelerated**: MFCC computation on CUDA for 10x speed improvement
- **Data Augmentation**: 
  - Pitch shifting (±1 semitone)
  - Additive white noise
  - SpecAugment-style frequency/time masking
- **Smart Padding/Truncation**: Fixed-length sequences (200 frames)

### 🧠 Model Capabilities
- **Convolutional Front-End**: Extracts spatial patterns from spectrograms
- **Bidirectional LSTM**: Captures temporal dependencies in both directions
- **Attention Pooling**: Focuses on emotionally salient frames
- **Regularization**: Dropout, label smoothing, weight decay
- **Stable Training**: Xavier initialization, gradient clipping

### 🚀 Performance Features
- **MFCC Caching**: Pre-computed features saved to `.npz` (runs once)
- **Mixed Precision**: Optional AMP for faster training
- **DataLoader Optimization**: Multi-worker prefetching with pinned memory
- **Early Stopping**: Prevents overfitting with patience monitoring
- **Checkpointing**: Saves best model during training

---

## 🏗 Architecture

```
Input Audio (WAV)
    ↓
MFCC Feature Extraction (40 coeffs)
    ↓
Delta & Delta-Delta Computation
    ↓
Normalization (per-feature)
    ↓
CNN Layers (2x Conv2D + MaxPool + Dropout)
    ↓
BiLSTM (2 layers, hidden_dim=192)
    ↓
Attention Mechanism (weighted pooling)
    ↓
Fully Connected Classifier
    ↓
6-Class Output (Softmax)
```

### Model Details

| Component | Configuration |
|-----------|--------------|
| **Input Shape** | (Batch, 120, 200) - 120 features × 200 time frames |
| **CNN Block 1** | Conv2D(1→32, 3×3) → ReLU → MaxPool(2×2) → Dropout(0.3) |
| **CNN Block 2** | Conv2D(32→64, 3×3) → ReLU → MaxPool(2×2) → Dropout(0.3) |
| **BiLSTM** | 2 layers, 192 hidden units, bidirectional, dropout=0.3 |
| **Attention** | Linear(384→128) → Tanh → Linear(128→1) → Softmax |
| **Classifier** | Linear(384→128) → ReLU → Dropout(0.5) → Linear(128→6) |
| **Parameters** | ~2.5M trainable parameters |

---

## 📊 Datasets

### Supported Datasets

#### 1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Location**: `data/RAVDESS/Audio_Speech_Actors_01-24/`
- **Size**: 1,440 speech files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprised (7 emotions)
- **Format**: 16-bit WAV, 48kHz (resampled to 16kHz)
- **Naming Convention**: `03-01-06-01-02-01-12.wav`
  - Position 3 (emotion): 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust

#### 2. **CREMA-D** (Crowd-Sourced Emotional Multimodal Actors Dataset)
- **Location**: `data/CREMA-D/AudioWAV/`
- **Size**: 7,442 speech files
- **Actors**: 91 actors (48 male, 43 female, diverse ethnicities)
- **Emotions**: Neutral, Happy, Sad, Angry, Fearful, Disgust (6 emotions)
- **Format**: 16-bit WAV, 16kHz
- **Naming Convention**: `1001_DFA_ANG_XX.wav`
  - ANG=Angry, DIS=Disgust, FEA=Fear, HAP=Happy, NEU=Neutral, SAD=Sad

### Dataset Statistics

| Dataset | Files | Emotions | Actors | Avg Duration |
|---------|-------|----------|--------|--------------|
| RAVDESS | 1,440 | 7 (6 used) | 24 | 3-5 sec |
| CREMA-D | 7,442 | 6 | 91 | 2-4 sec |
| **Total** | **~8,882** | **6** | **115** | **3 sec** |

---

## 💻 System Requirements

### Recommended Hardware (Optimized Configuration)

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **CPU** | AMD Ryzen 5 5600X (6-core/12-thread) or equivalent | Multi-threaded data loading |
| **RAM** | 32GB DDR4 @ 3600MHz | MFCC caching + DataLoader prefetching |
| **GPU** | NVIDIA RTX 3060 Ti (8GB VRAM) or better | Neural network training |
| **Storage** | 20GB free space (SSD recommended) | Datasets + cache files |

### Minimum Requirements

- **CPU**: 4-core processor
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA GTX 1660 (6GB VRAM) or equivalent
- **Storage**: 15GB free space

### Software Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **cuDNN**: 8.6+ (bundled with PyTorch)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rohan1011/SpeechModel.git
cd SpeechModel
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify CUDA Installation

```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

### 5. Download Datasets

#### RAVDESS
1. Visit: https://zenodo.org/record/1188976
2. Download: "Audio-only files" (Speech channel)
3. Extract to: `data/RAVDESS/Audio_Speech_Actors_01-24/`

#### CREMA-D
1. Visit: https://github.com/CheyneyComputerScience/CREMA-D
2. Download: AudioWAV.zip
3. Extract to: `data/CREMA-D/AudioWAV/`

### 6. Verify Dataset Structure

```
Speech-emotion/
├── data/
│   ├── RAVDESS/
│   │   └── Audio_Speech_Actors_01-24/
│   │       ├── Actor_01/
│   │       ├── Actor_02/
│   │       └── ... (24 folders)
│   └── CREMA-D/
│       └── AudioWAV/
│           ├── 1001_DFA_ANG_XX.wav
│           ├── 1001_DFA_DIS_XX.wav
│           └── ... (~7,442 files)
```

---

## 📁 Project Structure

```
Speech-emotion/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 train_emotion_rnn.py              # Main training script
│
├── 📂 data/                             # Audio datasets (not in repo)
│   ├── RAVDESS/
│   │   └── Audio_Speech_Actors_01-24/
│   └── CREMA-D/
│       └── AudioWAV/
│
├── 📂 features/                         # Cached MFCC features
│   └── mfcc_cache_cnn_bilstm_stable.npz
│
├── 📂 models/                           # Saved model checkpoints
│   ├── emotion_cnn_bilstm_stable_best.pth
│   └── emotion_cnn_bilstm_stable_checkpoint.pth
│
└── 📂 outputs/                          # Training outputs
    └── confusion_cnn_bilstm_stable.png
```

---

## ⚙️ Configuration

### Key Hyperparameters

Edit these in `train_emotion_rnn.py`:

```python
# Audio Processing
SAMPLE_RATE = 16000          # Audio sampling rate (Hz)
N_MFCC = 40                  # Number of MFCC coefficients
MAX_LEN = 200                # Sequence length (frames)

# Training
BATCH_SIZE = 64              # Batch size (adjust for your GPU)
NUM_EPOCHS = 120             # Maximum training epochs
LEARNING_RATE = 5e-5         # Initial learning rate
WEIGHT_DECAY = 1e-4          # L2 regularization strength

# Model Architecture
HIDDEN_DIM = 192             # LSTM hidden dimension
NUM_LSTM_LAYERS = 2          # Number of LSTM layers
LSTM_DROPOUT = 0.3           # LSTM dropout rate
CLASSIFIER_DROPOUT = 0.5     # Final classifier dropout

# System
NUM_WORKERS = 4              # DataLoader worker threads
DEVICE = "cuda"              # Training device (cuda/cpu)
```

### Hardware-Specific Tuning

#### For RTX 3060 Ti (8GB VRAM)
```python
BATCH_SIZE = 64              # Sweet spot for 8GB
NUM_WORKERS = 8              # Use 8 of 12 CPU threads
```

#### For RTX 3080/3090 (10-24GB VRAM)
```python
BATCH_SIZE = 128             # Larger batches possible
NUM_WORKERS = 12             # More workers if CPU allows
HIDDEN_DIM = 256             # Larger model capacity
```

#### For GTX 1660 (6GB VRAM)
```python
BATCH_SIZE = 32              # Reduced batch size
NUM_WORKERS = 4              # Conservative worker count
HIDDEN_DIM = 128             # Smaller model
```

#### For CPU-Only Training
```python
BATCH_SIZE = 16              # Small batches
NUM_WORKERS = 4              # Match CPU cores
DEVICE = "cpu"               # Force CPU mode
```

---

## 🚀 Usage

### Basic Training

```bash
python train_emotion_rnn.py
```

### Training Output

```
========== GPU Diagnostics ==========
CUDA available: True
GPU in use: NVIDIA GeForce RTX 3060 Ti
CUDA version: 11.8
PyTorch version: 2.1.0
=====================================

🎵 Loading dataset files...
📂 Scanning: data/RAVDESS/Audio_Speech_Actors_01-24
   ✅ Found 1440 files in Audio_Speech_Actors_01-24
📂 Scanning: data/CREMA-D/AudioWAV
   ✅ Found 7442 files in AudioWAV
✅ Found 8882 audio files.

📦 Loading cached MFCCs from mfcc_cache_cnn_bilstm_stable.npz...
✅ Loaded 8882 cached MFCCs (1256.3 MB)

📊 Train samples: 7105, Test samples: 1777

🚀 Training model (Stable CNN + BiLSTM + Attention) ...

Epoch [1/120]  Loss: 1.6243  Accuracy: 32.45%
Epoch [2/120]  Loss: 1.3821  Accuracy: 45.67%
  ✅ Best model saved.
...
Epoch [48/120]  Loss: 0.2134  Accuracy: 92.38%
  ✅ Best model saved.
⏹ Early stopping: no improvement for 12 epochs.

📈 Evaluating model...

========== Classification Report ==========

              precision    recall  f1-score   support

     neutral      0.912     0.934     0.923       342
       happy      0.885     0.867     0.876       289
         sad      0.941     0.928     0.934       298
       angry      0.893     0.901     0.897       312
     fearful      0.878     0.891     0.884       276
     disgust      0.901     0.885     0.893       260

    accuracy                          0.905      1777
   macro avg      0.902     0.901     0.901      1777
weighted avg      0.905     0.905     0.905      1777

📊 Confusion matrix saved to confusion_cnn_bilstm_stable.png
✅ Overall Accuracy: 90.48%

✅ Checkpoint saved as emotion_cnn_bilstm_stable_checkpoint.pth
```

### Inference / Prediction

To use the trained model for prediction on new audio files:

```python
import torch
import torchaudio
from train_emotion_rnn import EmotionCNNBiLSTM, extract_mfcc_with_deltas, EMOTIONS_MAP

# Load model
model = EmotionCNNBiLSTM()
model.load_state_dict(torch.load("emotion_cnn_bilstm_stable_best.pth"))
model.eval()
model.to("cuda")

# Process audio
audio_file = "path/to/your/audio.wav"
mfcc = extract_mfcc_with_deltas(audio_file, augment=False)
mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to("cuda")

# Predict
with torch.no_grad():
    output = model(mfcc_tensor)
    probs = torch.softmax(output, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

# Get emotion label
emotion_labels = {v: k for k, v in EMOTIONS_MAP.items()}
predicted_emotion = emotion_labels[pred_idx]
confidence = probs[0][pred_idx].item()

print(f"Predicted Emotion: {predicted_emotion} (confidence: {confidence:.2%})")
```

---

## 🎓 Training Details

### Training Process

1. **Data Loading**: Scans RAVDESS and CREMA-D directories
2. **MFCC Pre-computation**: Extracts and caches features (runs once)
3. **Train/Test Split**: 80/20 stratified split (maintains class balance)
4. **Model Initialization**: Xavier initialization for stable gradients
5. **Training Loop**: 
   - Forward pass with mixed precision (optional)
   - Cross-entropy loss with label smoothing (0.01)
   - Backward pass with gradient clipping (max_norm=0.5)
   - AdamW optimizer step with weight decay
6. **Learning Rate Scheduling**: ReduceLROnPlateau (halves LR after 6 epochs without improvement)
7. **Early Stopping**: Stops if validation loss doesn't improve for 12 epochs
8. **Evaluation**: Classification report and confusion matrix

### Optimization Techniques

| Technique | Purpose | Configuration |
|-----------|---------|---------------|
| **Label Smoothing** | Prevents overconfidence | ε = 0.01 |
| **Weight Decay** | L2 regularization | 1e-4 |
| **Gradient Clipping** | Prevents exploding gradients | max_norm = 0.5 |
| **Dropout** | Prevents overfitting | LSTM: 0.3, Classifier: 0.5 |
| **Xavier Init** | Stable gradient flow | Applied to Conv2D & Linear |
| **Data Augmentation** | Increases diversity | Pitch shift, noise, masking |
| **Early Stopping** | Prevents overfitting | Patience = 12 epochs |
| **LR Scheduling** | Adaptive learning rate | ReduceLROnPlateau |

### Memory Management

- **MFCC Caching**: Pre-computed features (~1.2GB) loaded into RAM
- **Pin Memory**: Faster CPU→GPU transfer
- **Persistent Workers**: Reduces worker restart overhead
- **Gradient Accumulation**: Optional for larger effective batch sizes

---

## 🏆 Performance

### Expected Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 88-92% |
| **Training Time** | 30-45 minutes (RTX 3060 Ti) |
| **Inference Time** | ~10ms per sample |
| **F1-Score (macro)** | 0.88-0.91 |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Neutral | 0.91 | 0.93 | 0.92 |
| Happy | 0.88 | 0.87 | 0.88 |
| Sad | 0.94 | 0.93 | 0.94 |
| Angry | 0.89 | 0.90 | 0.90 |
| Fearful | 0.88 | 0.89 | 0.88 |
| Disgust | 0.90 | 0.88 | 0.89 |

### Confusion Matrix

![Confusion Matrix](outputs/confusion_cnn_bilstm_stable.png)

*(Generated after training)*

---

## 🛠 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
- Reduce `BATCH_SIZE` (try 32, 16, or 8)
- Reduce `HIDDEN_DIM` (try 128 or 96)
- Reduce `NUM_WORKERS` (try 2 or 0)
- Close other GPU applications
- Use mixed precision training (already enabled)

#### 2. DataLoader Worker Errors (Windows)

**Error:**
```
RuntimeError: DataLoader worker exited unexpectedly
```

**Solutions:**
- Set `NUM_WORKERS = 0` (single-process loading)
- Wrap main code in `if __name__ == "__main__":`
- Set `persistent_workers=False`

#### 3. FileNotFoundError: Dataset Not Found

**Error:**
```
⚠️ Directory not found: data/RAVDESS/...
```

**Solutions:**
- Verify dataset extraction paths match `DATA_DIRS`
- Check folder names (case-sensitive on Linux)
- Ensure datasets are downloaded completely

#### 4. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'torchaudio'
```

**Solutions:**
```bash
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Low Accuracy / Poor Convergence

**Symptoms:** Accuracy stuck at ~20-30%

**Solutions:**
- Check dataset labels are parsed correctly
- Verify MFCC cache isn't corrupted (delete `.npz` file)
- Increase `LEARNING_RATE` to 1e-4
- Ensure GPU is being used (`DEVICE = cuda`)
- Try training longer (increase `NUM_EPOCHS`)

#### 6. Slow Training

**Symptoms:** <1 epoch per 10 minutes

**Solutions:**
- Verify GPU is active (check `show_gpu_info()` output)
- Increase `NUM_WORKERS` (try 8-12 on Ryzen 5 5600X)
- Enable `pin_memory=True` in DataLoader
- Check for CPU bottleneck (Task Manager / `htop`)
- Ensure data is on SSD, not HDD

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement

- [ ] Add transformer-based models (Wav2Vec2, HuBERT)
- [ ] Implement real-time emotion detection
- [ ] Add more datasets (IEMOCAP, EmoDB, SAVEE)
- [ ] Create web interface with Gradio/Streamlit
- [ ] Optimize for mobile deployment (ONNX, TFLite)
- [ ] Multi-language support
- [ ] Gender/age-invariant features

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Rohan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Datasets
- **RAVDESS**: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE 13(5): e0196391*. https://doi.org/10.1371/journal.pone.0196391
- **CREMA-D**: Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377-390.

### Frameworks & Libraries
- **PyTorch**: Paszke et al. (2019) - Deep learning framework
- **torchaudio**: Audio processing for PyTorch
- **scikit-learn**: Machine learning utilities
- **librosa**: Audio analysis tools
- **matplotlib & seaborn**: Visualization

### Inspiration
- Research papers on emotion recognition
- PyTorch community tutorials
- Open-source SER projects

---

## 📞 Contact

**Project Maintainer**: Rohan  
**GitHub**: [@Rohan1011](https://github.com/Rohan1011)  
**Repository**: [SpeechModel](https://github.com/Rohan1011/SpeechModel)

---

## 📚 References

1. Livingstone SR, Russo FA (2018). "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)." *PLoS ONE*.
2. Cao H et al. (2014). "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset." *IEEE Trans. Affective Computing*.
3. Hochreiter S, Schmidhuber J (1997). "Long Short-Term Memory." *Neural Computation*.
4. Bahdanau D et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*.
5. Davis S, Mermelstein P (1980). "Comparison of Parametric Representations for Monosyllabic Word Recognition." *IEEE Trans. ASSP*.

---

## 🎯 Roadmap

### Version 1.0 (Current)
- ✅ CNN + BiLSTM architecture
- ✅ RAVDESS + CREMA-D support
- ✅ MFCC feature extraction
- ✅ GPU acceleration
- ✅ Training pipeline

### Version 1.1 (Planned)
- [ ] Inference script with CLI
- [ ] Model quantization (INT8)
- [ ] ONNX export
- [ ] Docker containerization

### Version 2.0 (Future)
- [ ] Transformer-based models
- [ ] Real-time audio stream processing
- [ ] REST API deployment
- [ ] Web UI (Gradio/Streamlit)
- [ ] Mobile app (Flutter)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ using PyTorch and Python

**Author**: Rohan Reddy  
🎓 Computer Science Student | 💻 AI & Web Dev Enthusiast

</div>

## 🧩 Project Structure

```
speech_emotion_recognition/
├── data/
│   ├── RAVDESS/
│   │   └── Audio_Speech_Actors_01-24/
│   ├── CREMA-D/
│   │   └── AudioWAV/
│   └── merged_metadata.csv
│
├── features/
│   ├── mfcc_train.npy
│   └── mfcc_test.npy
│
├── models/
│   └── emotion_rnn.py
│
├── train_emotion_rnn_safe.py
├── train_emotion_rnn_gpu_amp.py
├── predict.py
└── README.md
```

---

## ⚙️ Environment Setup

### Requirements
- Python ≥ 3.10  
- PyTorch ≥ 2.5.0 (CUDA 12.x build)  
- Torchaudio (matching PyTorch version)  
- scikit-learn, matplotlib, seaborn, numpy

### Installation

```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn matplotlib seaborn tqdm
```

---

## 🎧 Datasets

### [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
### [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)

> Place both datasets inside `speech_emotion_recognition/data/` as shown above.

---

## 🧠 Model Architecture

### EmotionRNN (Bidirectional LSTM)
| Component | Description |
|------------|-------------|
| Input | MFCC feature vectors (40 × 200) |
| Encoder | 2-layer bidirectional LSTM (128 hidden units) |
| Classifier | Fully connected layer (256 → 6 classes) |
| Output | Emotion logits (`[batch, 6]`) |

---

## 🧩 Training Modes

### 🛡️ Safe Mode
```bash
python train_emotion_rnn_safe.py
```

### ⚡ GPU High-Performance Mode
```bash
python train_emotion_rnn_gpu_amp.py
```

---

## 🧰 Configurable Parameters

```python
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
```

---

## 🧮 Training Output

Model shows epoch losses and a confusion matrix with classification metrics.

---

## 💾 Model Saving

Trained weights are saved as:
```
emotion_rnn_gpu_safe.pth
```

---

## 🧠 Inference (predict.py)

```bash
python predict.py --file path/to/audio.wav
```

Output example:
```
Predicted Emotion: Happy (confidence: 92.3%)
```

---

## ⚡ Performance Tips

| Area | Action | Effect |
|-------|---------|---------|
| DataLoader | Increase `num_workers` | Uses more CPU cores |
| MFCC Extraction | Use GPU | 3× faster |
| Batch Size | 64–128 | Better throughput |
| Mixed Precision | Keep enabled | Saves VRAM |
| Disk I/O | Use SSD | Faster data access |

---

## 🧩 Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| CUDA OOM | Batch too large | Reduce `BATCH_SIZE` |
| KeyboardInterrupt | Too many workers | Use `num_workers=0` |
| Repeated GPU Diagnostics | Script re-runs | Use `if __name__ == "__main__"` |
| MFCC warning | Too many Mel bins | Use `n_mels=64` |

---

## 🧰 Hardware Recommendations

| Component | Minimum            | Ideal                     |
|-----------|--------------------|---------------------------|
| GPU       | GTX 1080ti/RTX 2060| RTX 3060 Ti / 4070        |
| VRAM      | 6 GB               | 8–12 GB                   |
| CPU       | Quad-core          | Ryzen 5 5600X / i5-12600K |
| RAM       | 16 GB              | 32 GB                     |
| Storage   | SSD                | NVMe SSD                  |

---

## 🧠 Future Enhancements
- [ ] Add attention mechanism  
- [ ] Add CNN-RNN hybrid model  
- [ ] Real-time inference (mic input)  
- [ ] Streamlit web app demo  

---

## 🏁 Author
**Rohan Reddy**  
🎓 Computer Science Student | 💻 AI & Web Dev Enthusiast  
GitHub: [Rohan1011](https://github.com/Rohan1011)
