# predict.py
"""
Speech Emotion Recognition — Prediction Utility
Usage:
    python predict.py --file path/to/audio.wav [--model emotion_wav2vec_bilstm_checkpoint.pth]
"""

import argparse
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

# =================== CONFIG ===================
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS_MAP = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgust"
}

# =================== MODEL DEFINITIONS ===================
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(out * attn, dim=1)
        return self.classifier(context)

# Optional: CNN + BiLSTM for MFCC model
class EmotionCNNBiLSTM(nn.Module):
    def __init__(self, n_mfcc=40, hidden_dim=192, num_layers=2, num_classes=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)), nn.Dropout(0.3),
            nn.Conv2d(32, 64, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)), nn.Dropout(0.3)
        )
        self.bilstm = None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.Tanh(), nn.Linear(128, 1, bias=False)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b = x.size(0)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        if self.bilstm is None:
            self.bilstm = nn.LSTM(c * f, self.hidden_dim, self.num_layers,
                                  batch_first=True, dropout=0.3, bidirectional=True).to(x.device)
        out, _ = self.bilstm(x)
        attn = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(out * attn, dim=1)
        return self.fc(context)

# =================== FEATURE EXTRACTORS ===================
def extract_wav2vec_features(file_path, max_len=400):
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(DEVICE)
    model.eval()
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(DEVICE)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
        feat = features[-1].squeeze(0).cpu().numpy()
    if feat.shape[0] < max_len:
        pad = np.zeros((max_len - feat.shape[0], feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    else:
        feat = feat[:max_len, :]
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

def extract_mfcc_features(file_path, n_mfcc=40, max_len=200):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc, melkwargs={"n_mels": 64}
    )
    with torch.no_grad():
        mfcc = mfcc_transform(waveform)
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        mfcc_full = torch.cat([mfcc, delta, delta2], dim=1).squeeze(0)
    if mfcc_full.shape[1] < max_len:
        pad = max_len - mfcc_full.shape[1]
        mfcc_full = torch.nn.functional.pad(mfcc_full, (0, pad))
    else:
        mfcc_full = mfcc_full[:, :max_len]
    mfcc_full = (mfcc_full - mfcc_full.mean(dim=1, keepdim=True)) / (
        mfcc_full.std(dim=1, keepdim=True) + 1e-8
    )
    return mfcc_full.unsqueeze(0)

# =================== PREDICTION LOGIC ===================
def predict_emotion(file_path, model_path):
    print(f"\n🎵 Analyzing: {file_path}\n")

    # decide model type
    is_wav2vec = "wav2vec" in os.path.basename(model_path).lower()
    print(f"🔍 Detected model type: {'Wav2Vec2 + BiLSTM' if is_wav2vec else 'MFCC + CNN + BiLSTM'}")

    # load model
    model = BiLSTM_Attention() if is_wav2vec else EmotionCNNBiLSTM()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint["model"])
    model = model.to(DEVICE)
    model.eval()

    # feature extraction
    features = extract_wav2vec_features(file_path) if is_wav2vec else extract_mfcc_features(file_path)
    features = features.to(DEVICE)

    with torch.no_grad():
        outputs = model(features)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    emotion = EMOTIONS_MAP[pred.item()]
    confidence = conf.item() * 100
    print(f"🎯 Predicted Emotion: {emotion.upper()} ({confidence:.2f}% confidence)\n")

# =================== ENTRY POINT ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .wav file")
    parser.add_argument("--model", type=str, default="emotion_wav2vec_bilstm_checkpoint.pth", help="Checkpoint path")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("❌ Error: Audio file not found!")
        exit(1)
    if not os.path.exists(args.model):
        print("❌ Error: Model checkpoint not found!")
        exit(1)

    predict_emotion(args.file, args.model)
