# 🎙️ Speech Emotion Recognition using RNNs (PyTorch + Torchaudio)

> End-to-end Speech Emotion Recognition (SER) model using **MFCC feature extraction** and **Bidirectional LSTMs (RNNs)**, built with **PyTorch** and **Torchaudio**.  
> Supports **GPU acceleration**, **mixed precision (AMP)**, and **data augmentation** for robust affect-aware applications.

---

## 🚀 Overview

This project trains a model to **classify emotions from speech audio** using the **RAVDESS** and **CREMA-D** datasets.  
It extracts **MFCC features**, feeds them into an **RNN (LSTM)** network, and outputs predicted emotional states such as:

> Neutral • Happy • Sad • Angry • Fearful • Disgust

---

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
