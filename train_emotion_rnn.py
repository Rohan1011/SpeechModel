# train_emotion_cnn_bilstm_stable.py
"""
Stable CNN + BiLSTM training script for Speech Emotion Recognition.
Features:
 - MFCC + delta + delta-delta (normalized)
 - CNN front-end -> BiLSTM -> attention pooling -> classifier
 - MFCC caching to .npz
 - Gradient clipping, weight init, low LR
 - Early stopping + ReduceLROnPlateau
Designed for RTX 3060 Ti (8GB) + Ryzen 5 5600X; adjust BATCH_SIZE/NUM_WORKERS as needed.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== CONFIG ========================
DATA_DIRS = ["data/RAVDESS/Audio_Speech_Actors_01-24", "data/CREMA-D/AudioWAV"]
EMOTIONS_MAP = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fearful': 4, 'disgust': 5}

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 200              # frames
BATCH_SIZE = 64            # safe default for 8GB VRAM; lower if OOM
NUM_EPOCHS = 120
LEARNING_RATE = 5e-5       # lowered to avoid instability
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 192
NUM_LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
CLASSIFIER_DROPOUT = 0.5

CACHE_FILE = "mfcc_cache_cnn_bilstm_stable.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4            # on Windows keep <=4; increase on Linux if you have spare cores

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================= GPU INFO =================
def show_gpu_info():
    print("========== GPU Diagnostics ==========")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU in use:", torch.cuda.get_device_name(0))
        except Exception:
            pass
        print("CUDA version:", torch.version.cuda)
        print("PyTorch version:", torch.__version__)
    print("=====================================\n")

# ======================== MFCC transform (on GPU) ========================
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_mels": 64}
).to(DEVICE)

# ======================== AUGMENTATION ========================
def augment_audio_waveform(waveform, sr):
    """Lightweight augmentations: pitch shift (small) or additive noise."""
    choice = random.choice(['none', 'pitch', 'noise'])
    try:
        if choice == 'pitch':
            n_steps = random.uniform(-1.0, 1.0)
            waveform = torchaudio.functional.pitch_shift(waveform, sr, n_steps)
        elif choice == 'noise':
            waveform = waveform + 0.001 * torch.randn_like(waveform)
    except Exception:
        # keep original on failure
        pass
    return waveform

# ======================== FEATURE EXTRACTION (with deltas + normalization) ========================
def extract_mfcc_with_deltas(file_path, augment=False):
    """
    Returns numpy array shape (3*N_MFCC, MAX_LEN) float32
    Normalizes per-feature after padding/truncation.
    """
    waveform, sr = torchaudio.load(file_path)  # [channels, samples]
    # convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(DEVICE)

    if augment:
        waveform = augment_audio_waveform(waveform, sr)

    with torch.no_grad():
        mfcc = mfcc_transform(waveform)  # [1, N_MFCC, T]
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        mfcc_full = torch.cat([mfcc, delta, delta2], dim=1)  # [1, 3*N_MFCC, T]

    mfcc_full = mfcc_full.squeeze(0).cpu().numpy()  # [3*N_MFCC, T]

    # pad/truncate time dimension to MAX_LEN
    if mfcc_full.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc_full.shape[1]
        mfcc_full = np.pad(mfcc_full, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_full = mfcc_full[:, :MAX_LEN]

    # normalize per-feature (after padding/truncation)
    mfcc_full = (mfcc_full - mfcc_full.mean(axis=1, keepdims=True)) / (
        mfcc_full.std(axis=1, keepdims=True) + 1e-8
    )

    return mfcc_full.astype(np.float32)

# ======================== PRE-COMPUTE MFCCs (cache) ========================
def precompute_mfccs(files, labels, cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        print(f"📦 Loading cached MFCCs from {cache_file}...")
        d = np.load(cache_file, allow_pickle=True)
        mfccs = d['mfccs']
        labs = d['labels']
        print(f"✅ Loaded {len(mfccs)} cached MFCCs ({mfccs.nbytes/1024**2:.1f} MB)\n")
        return mfccs, labs

    print(f"🔄 Pre-computing MFCCs for {len(files)} files (this runs once)...")
    mfccs = []
    valid_labels = []
    for i, (f, lab) in enumerate(zip(files, labels)):
        if i % 200 == 0:
            print(f"   Progress: {i}/{len(files)} ({100.0 * i / len(files):.1f}%)")
        try:
            arr = extract_mfcc_with_deltas(f, augment=False)
            mfccs.append(arr)
            valid_labels.append(lab)
        except Exception as e:
            print(f"   ⚠️ Skipping {f}: {e}")
    mfccs = np.stack(mfccs, axis=0)
    valid_labels = np.array(valid_labels, dtype=np.int64)
    np.savez_compressed(cache_file, mfccs=mfccs, labels=valid_labels)
    print(f"\n💾 Saved cache: {cache_file}  (items={len(mfccs)}, size={mfccs.nbytes/1024**2:.1f} MB)\n")
    return mfccs, valid_labels

# ======================== DATASET ========================
class MFCCDataset(Dataset):
    def __init__(self, mfccs, labels, augment=False):
        self.mfccs = mfccs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        x = self.mfccs[idx].copy()  # [3*N_MFCC, MAX_LEN]
        # simple SpecAugment-like random masking
        if self.augment and random.random() < 0.5:
            # time mask
            t0 = random.randint(0, max(1, x.shape[1] - 16))
            x[:, t0:t0 + random.randint(0, 16)] = 0
            # freq mask
            f0 = random.randint(0, max(1, x.shape[0] - 8))
            x[f0:f0 + random.randint(0, 8), :] = 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ======================== MODEL: CNN + BiLSTM + Attention ========================
class EmotionCNNBiLSTM(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, hidden_dim=HIDDEN_DIM, num_layers=NUM_LSTM_LAYERS, num_classes=len(EMOTIONS_MAP)):
        super().__init__()
        # input will be [B, 3*N_MFCC, MAX_LEN] -> we add channel dim -> [B,1,F,T]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.3)
        )
        self.bilstm = None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_dropout = LSTM_DROPOUT

        # attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, features (=3*N_MFCC), T]
        b = x.size(0)
        x = x.unsqueeze(1)            # [B, 1, F, T]
        x = self.cnn(x)               # [B, C, F', T']
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, T', feat_dim]

        # lazy LSTM init (depends on CNN output dims)
        if self.bilstm is None:
            self.bilstm = nn.LSTM(
                input_size=c * f,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.lstm_dropout,
                bidirectional=True
            ).to(x.device)

        out, _ = self.bilstm(x)  # out: [B, T', 2*hidden_dim]

        # attention pooling
        attn_logits = self.attention(out)                 # [B, T', 1]
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, T', 1]
        context = torch.sum(out * attn_weights, dim=1)    # [B, 2*hidden_dim]

        logits = self.classifier(context)                 # [B, num_classes]
        return logits

# ======================== HELPERS: parse filenames ========================
def parse_ravdess_emotion(filename):
    # filename pattern: '03-01-02-01-01-02-01.wav' -> the 3rd block is emotion code
    try:
        parts = filename.replace('.wav', '').split('-')
        if len(parts) >= 3:
            code = parts[2]
            mapping = {
                '01': EMOTIONS_MAP['neutral'],
                '02': EMOTIONS_MAP['neutral'],
                '03': EMOTIONS_MAP['happy'],
                '04': EMOTIONS_MAP['sad'],
                '05': EMOTIONS_MAP['angry'],
                '06': EMOTIONS_MAP['fearful'],
                '07': EMOTIONS_MAP['disgust']
            }
            return mapping.get(code, None)
    except Exception:
        return None
    return None

def parse_cremad_emotion(filename):
    n = filename.lower()
    if 'neu' in n: return EMOTIONS_MAP['neutral']
    if 'hap' in n: return EMOTIONS_MAP['happy']
    if 'sad' in n: return EMOTIONS_MAP['sad']
    if 'ang' in n: return EMOTIONS_MAP['angry']
    if 'fea' in n: return EMOTIONS_MAP['fearful']
    if 'dis' in n: return EMOTIONS_MAP['disgust']
    return None

def load_data_files():
    files, labels = [], []
    for dir_path in DATA_DIRS:
        if not os.path.exists(dir_path):
            print(f"⚠️ Directory not found: {dir_path}")
            continue
        print(f"📂 Scanning: {dir_path}")
        count = 0
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                if not f.endswith(".wav"):
                    continue
                path = os.path.join(root, f)
                label = None
                # simple detection by folder name
                if 'RAVDESS' in dir_path or 'Actor_' in root:
                    label = parse_ravdess_emotion(f)
                else:
                    label = parse_cremad_emotion(f)
                if label is not None:
                    files.append(path)
                    labels.append(label)
                    count += 1
        print(f"   ✅ Found {count} files in {os.path.basename(dir_path)}")
    return files, labels

# ======================== TRAIN / EVAL ========================
def train_model(model, loader, criterion, optimizer, scheduler, num_epochs, patience=12, clip_norm=0.5):
    model.train()
    best_loss = float('inf')
    wait = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for mfccs, labels in loader:
            mfccs = mfccs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        scheduler.step(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.2f}%")

        # early stopping logic
        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            wait = 0
            torch.save(model.state_dict(), "emotion_cnn_bilstm_stable_best.pth")
            print("  ✅ Best model saved.")
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping: no improvement for {patience} epochs.")
                break

def evaluate_model(model, test_loader, save_cm="confusion_cnn_bilstm_stable.png"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs = mfccs.to(DEVICE)
            outputs = model(mfccs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    labels_present = sorted(unique_labels(y_true + y_pred))
    label_names = [list(EMOTIONS_MAP.keys())[i] for i in labels_present]

    print("\n========== Classification Report ==========\n")
    print(classification_report(y_true, y_pred, labels=labels_present, target_names=label_names, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix — Stable CNN+BiLSTM")
    plt.tight_layout()
    plt.savefig(save_cm, dpi=200)
    print(f"\n📊 Confusion matrix saved to {save_cm}")
    plt.show()

    acc = np.trace(cm) / np.sum(cm)
    print(f"\n✅ Overall Accuracy: {acc*100:.2f}%\n")

# ======================== MAIN ========================
if __name__ == "__main__":
    show_gpu_info()

    print("🎵 Loading dataset files...")
    files, labels = load_data_files()
    print(f"✅ Found {len(files)} audio files.\n")

    # Precompute MFCCs (will cache)
    mfccs, labels = precompute_mfccs(files, labels, CACHE_FILE)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, stratify=labels, random_state=42)
    print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # datasets and loaders
    train_ds = MFCCDataset(X_train, y_train, augment=True)
    test_ds = MFCCDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=max(1, NUM_WORKERS//2), pin_memory=True, persistent_workers=False)

    # model, criterion, optimizer, scheduler
    model = EmotionCNNBiLSTM(n_mfcc=N_MFCC, hidden_dim=HIDDEN_DIM, num_layers=NUM_LSTM_LAYERS).to(DEVICE)

    # Xavier init for stable start
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    print("🚀 Training model (Stable CNN + BiLSTM + Attention) ...\n")
    train_model(model, train_loader, criterion, optimizer, scheduler, NUM_EPOCHS, patience=12)

    print("\n📈 Evaluating model...\n")
    evaluate_model(model, test_loader, save_cm="confusion_cnn_bilstm_stable.png")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'emotions_map': EMOTIONS_MAP
    }, "emotion_cnn_bilstm_stable_checkpoint.pth")
    print("✅ Checkpoint saved as emotion_cnn_bilstm_stable_checkpoint.pth")
