"""
Dynamic ISL Training — v8: v7 features (59) + on-the-fly augmentation

Augmentations (applied only to training batches):
    - Temporal jitter (drop/duplicate 1-2 frames)
    - Time warping (nonlinear resample)
    - Gaussian noise on all features
    - Angular noise on the angle block (indices 0-35)
    - Palm-normal noise (indices 36-41)
    - Frame masking (zero 1-2 random frames)
"""

import os
import sys
import glob
import random
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# IMPORT MODEL
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from dynamic_cnn import UnifiedISLNet

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
FEATURES_ROOT   = os.path.join(SCRIPT_DIR, "features")
TRAIN_DIR       = os.path.join(FEATURES_ROOT, "Train_CSV")
TEST_DIR        = os.path.join(FEATURES_ROOT, "TEST CSV")

OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "outputs_dynamic_v8")
CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_DIR      = os.path.join(OUTPUT_DIR, "history")
DATA_SPLIT_DIR   = os.path.join(OUTPUT_DIR, "data_splits")
MODEL_EXPORT_DIR = os.path.join(SCRIPT_DIR, "models_dynamic_v8")

for d in [CHECKPOINT_DIR, HISTORY_DIR, DATA_SPLIT_DIR, MODEL_EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MAX_TIME  = 60
FEATURES  = 59
BATCH_SIZE = 32
NUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

TRAIN_SUBJECTS = {"SRII", "SANJAY"}
TEST_SUBJECT   = "NANDY"
EXCLUDE_CLASSES = set()

# Feature block indices (for targeted augmentation)
IDX_ANGLES_L = slice(0, 18)
IDX_ANGLES_R = slice(18, 36)
IDX_PALM_L   = slice(36, 39)
IDX_PALM_R   = slice(39, 42)
IDX_H2H      = slice(42, 43)
IDX_MOTION   = slice(43, 59)

KEEP_INDICES_BASE = list(range(126, 168)) + [249]

# ============================================================
# SEEDING
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"DEVICE:   {DEVICE}")
print(f"FEATURES: {FEATURES}  (v8: v7 + on-the-fly augmentation)")
print(f"EXCLUDED: {sorted(EXCLUDE_CLASSES) if EXCLUDE_CLASSES else 'none'}")
print("=" * 70)


# ============================================================
# MOTION FEATURES (same as v7)
# ============================================================
def compute_motion_features(arr):
    T = arr.shape[0]
    motion = np.zeros((T, 16), dtype=np.float32)

    l_wrist = arr[:, 0:3]
    r_wrist = arr[:, 63:66]

    l_vel = np.zeros_like(l_wrist)
    r_vel = np.zeros_like(r_wrist)
    l_vel[1:] = l_wrist[1:] - l_wrist[:-1]
    r_vel[1:] = r_wrist[1:] - r_wrist[:-1]

    l_acc = np.zeros_like(l_vel)
    r_acc = np.zeros_like(r_vel)
    l_acc[1:] = l_vel[1:] - l_vel[:-1]
    r_acc[1:] = r_vel[1:] - r_vel[:-1]

    l_angle = np.zeros(T, dtype=np.float32)
    r_angle = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        l_angle[t] = float(np.degrees(np.arctan2(l_vel[t, 1], l_vel[t, 0])))
        r_angle[t] = float(np.degrees(np.arctan2(r_vel[t, 1], r_vel[t, 0])))

    l_curv = np.zeros(T, dtype=np.float32)
    r_curv = np.zeros(T, dtype=np.float32)
    eps = 1e-6
    for t in range(2, T):
        a = l_vel[t]; b = l_vel[t - 1]
        l_curv[t] = float(np.linalg.norm(np.cross(a, b)) /
                          ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))
        a = r_vel[t]; b = r_vel[t - 1]
        r_curv[t] = float(np.linalg.norm(np.cross(a, b)) /
                          ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))

    motion[:, 0:3]  = l_vel
    motion[:, 3:6]  = r_vel
    motion[:, 6:9]  = l_acc
    motion[:, 9:12] = r_acc
    motion[:, 12]   = l_angle
    motion[:, 13]   = r_angle
    motion[:, 14]   = l_curv
    motion[:, 15]   = r_curv
    return motion


# ============================================================
# TEMPORAL RESAMPLING
# ============================================================
def resample_clip(clip, target_len=MAX_TIME):
    T, F = clip.shape
    if T == target_len:
        return clip
    if T > target_len:
        idx = np.linspace(0, T - 1, target_len).astype(int)
        return clip[idx]
    pad = np.zeros((target_len - T, F), dtype=clip.dtype)
    return np.vstack([clip, pad])


# ============================================================
# AUGMENTATION — applied per-sample on the fly
# ============================================================
def augment_sequence(seq, rng):
    """
    seq: (T, 59) float32 already scaled.
    Returns: (T, 59) augmented copy.
    """
    seq = seq.copy()
    T = seq.shape[0]

    # ---- 1. Temporal jitter (drop 1-2 frames, duplicate others) ----
    if rng.random() < 0.5:
        n_drop = rng.integers(1, 3)
        for _ in range(n_drop):
            i = rng.integers(1, T - 1)
            seq = np.delete(seq, i, axis=0)
            j = rng.integers(0, len(seq))
            seq = np.insert(seq, j, seq[j], axis=0)
        seq = seq[:T]

    # ---- 2. Time warp (nonlinear resample) ----
    if rng.random() < 0.5:
        t_old = np.linspace(0, 1, T)
        t_new = np.sort(np.clip(t_old + rng.normal(0, 0.03, T), 0, 1))
        seq = np.array([np.interp(t_new, t_old, seq[:, k])
                        for k in range(seq.shape[1])]).T.astype(np.float32)

    # ---- 3. Gaussian noise on all features ----
    if rng.random() < 0.7:
        seq = seq + rng.normal(0, 0.02, seq.shape).astype(np.float32)

    # ---- 4. Angular noise (targets the angle block) ----
    if rng.random() < 0.5:
        seq[:, IDX_ANGLES_L] = seq[:, IDX_ANGLES_L] + rng.normal(0, 0.05, (T, 18))
        seq[:, IDX_ANGLES_R] = seq[:, IDX_ANGLES_R] + rng.normal(0, 0.05, (T, 18))

    # ---- 5. Palm-normal noise (targets the palm block) ----
    if rng.random() < 0.5:
        seq[:, IDX_PALM_L] = seq[:, IDX_PALM_L] + rng.normal(0, 0.03, (T, 3))
        seq[:, IDX_PALM_R] = seq[:, IDX_PALM_R] + rng.normal(0, 0.03, (T, 3))

    # ---- 6. Frame masking (zero 1-2 random frames) ----
    if rng.random() < 0.3:
        n_mask = rng.integers(1, 3)
        for _ in range(n_mask):
            i = rng.integers(0, T)
            seq[i] = 0.0

    return seq.astype(np.float32)


# ============================================================
# DATASET  (⭐ corrected — __getitem__ inside the class)
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, X, y, augment=False, seed=SEED):
        self.X = X.astype(np.float32)      # (N, T, F)
        self.y = y.astype(np.int64)        # (N,)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]
        if self.augment:
            seq = augment_sequence(seq, self.rng)
        # (T, F) → (1, T, F) to add channel dim for Conv2D
        return torch.from_numpy(seq).unsqueeze(0), torch.tensor(self.y[idx])


# ============================================================
# CSV DISCOVERY
# ============================================================
def discover_csvs(root_dir):
    entries = []
    if not os.path.isdir(root_dir):
        print(f"⚠️  Folder not found: {root_dir}")
        return entries
    for class_dir in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        if class_dir in EXCLUDE_CLASSES:
            continue
        label = class_dir
        for csv_path in sorted(glob.glob(os.path.join(class_path, "*.csv"))):
            subject = os.path.basename(csv_path).split("_")[0]
            entries.append((csv_path, subject, label))
    return entries


# ============================================================
# LOAD CLIPS
# ============================================================
def load_clips(entries, split_name):
    clips, labels, subjects = [], [], []
    stats = {"unchanged": 0, "resampled": 0, "padded": 0}

    for csv_path, subject, label in entries:
        df = pd.read_csv(csv_path)
        if "frame" in df.columns:
            df = df.drop(columns=["frame"])
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        arr = df.values.astype(np.float32)
        if arr.shape[1] < 252:
            continue

        base = arr[:, KEEP_INDICES_BASE]        # (T, 43)
        motion = compute_motion_features(arr)   # (T, 16)
        feat = np.concatenate([base, motion], axis=1)   # (T, 59)

        T = feat.shape[0]
        feat = resample_clip(feat, MAX_TIME)

        if T == MAX_TIME:   stats["unchanged"] += 1
        elif T > MAX_TIME:  stats["resampled"] += 1
        else:               stats["padded"]    += 1

        clips.append(feat)
        labels.append(label)
        subjects.append(subject)

    if not clips:
        print(f"❌ No clips for {split_name}.")
        return None, None, None

    X = np.stack(clips, axis=0)
    y = np.array(labels)
    s = np.array(subjects)
    print(f"\n[{split_name}] Loaded {len(clips)} clips")
    print(f"  Unchanged: {stats['unchanged']} | Resampled: {stats['resampled']} | Padded: {stats['padded']}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes ({len(set(y))}): {sorted(set(y))}")
    print(f"  Subjects: {sorted(set(s))}")
    return X, y, s


# ============================================================
# DISCOVER + LOAD
# ============================================================
print("\n" + "=" * 70)
print("DISCOVERING DATASET")
print("=" * 70)

train_entries = discover_csvs(TRAIN_DIR)
test_entries  = discover_csvs(TEST_DIR)

X_train_pool, y_train_pool, s_train_pool = load_clips(train_entries, "TRAIN/VAL POOL")
X_test,       y_test,       s_test       = load_clips(test_entries,  "TEST")

if X_train_pool is None or X_test is None:
    print("\n❌ Missing data. Exiting.")
    sys.exit(1)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"\nTRAIN/VALIDATION: {len(X_train_pool)} clips")
for subj in sorted(set(s_train_pool)):
    n = int((s_train_pool == subj).sum())
    print(f"  {subj}: {n} clips")
print(f"TEST: {len(X_test)} clips (subject {TEST_SUBJECT})")

assert set(s_train_pool).issubset(TRAIN_SUBJECTS)
assert set(s_test).issubset({TEST_SUBJECT})
assert len(set(s_train_pool) & set(s_test)) == 0
print("\n✅ No subject leakage.")

# ============================================================
# SPLIT + ENCODE + SCALE
# ============================================================
idx = np.arange(len(X_train_pool))
tr_idx, va_idx = train_test_split(idx, test_size=0.15,
                                  random_state=SEED, stratify=y_train_pool)
X_train, y_train = X_train_pool[tr_idx], y_train_pool[tr_idx]
X_val,   y_val   = X_train_pool[va_idx], y_train_pool[va_idx]

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc   = encoder.transform(y_val)
y_test_enc  = encoder.transform(y_test)
joblib.dump(encoder, os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"))
print(f"\nClasses ({len(encoder.classes_)}): {list(encoder.classes_)}")

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, FEATURES))
X_train_s = scaler.transform(X_train.reshape(-1, FEATURES)).reshape(-1, MAX_TIME, FEATURES)
X_val_s   = scaler.transform(X_val.reshape(-1, FEATURES)).reshape(-1, MAX_TIME, FEATURES)
X_test_s  = scaler.transform(X_test.reshape(-1, FEATURES)).reshape(-1, MAX_TIME, FEATURES)
joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
print("Scaler fit on TRAIN only.")

# ============================================================
# SUBJECT DISTANCE CHECK
# ============================================================
print("\n" + "=" * 70)
print("SUBJECT DISTANCE CHECK")
print("=" * 70)

subj_means = {}
for subj in set(s_train_pool) | set(s_test):
    if subj in set(s_train_pool):
        subj_means[subj] = X_train_pool[s_train_pool == subj].mean(axis=(0, 1))
    else:
        subj_means[subj] = X_test[s_test == subj].mean(axis=(0, 1))

subjects = sorted(subj_means.keys())
print(f"\n{'':10s}" + "".join(f"{s:>12s}" for s in subjects))
for s1 in subjects:
    row = f"{s1:<10s}"
    for s2 in subjects:
        d = np.linalg.norm(subj_means[s1] - subj_means[s2])
        row += f"{d:>12.3f}"
    print(row)

# ============================================================
# DATASETS / LOADERS
# ============================================================
train_ds = SequenceDataset(X_train_s, y_train_enc, augment=True,  seed=SEED)
val_ds   = SequenceDataset(X_val_s,   y_val_enc,   augment=False)
test_ds  = SequenceDataset(X_test_s,  y_test_enc,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# MODEL
# ============================================================
model = UnifiedISLNet(num_classes=len(encoder.classes_),
                      max_time=MAX_TIME, features=FEATURES).to(DEVICE)
print(f"\nModel: {model.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# LOSS / OPT / SCHED
# ============================================================
cw = compute_class_weight("balanced", classes=np.unique(y_train_enc), y=y_train_enc)
cw_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw_t)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                        factor=0.5, patience=3)

LATEST = os.path.join(CHECKPOINT_DIR, "latest.pth")
BEST   = os.path.join(CHECKPOINT_DIR, "best_model.pth")
FINAL  = os.path.join(CHECKPOINT_DIR, "final_model.pth")
FULL   = os.path.join(MODEL_EXPORT_DIR, "full_model.pth")

# ============================================================
# RESUME
# ============================================================
FINGERPRINT = {"features": FEATURES, "max_time": MAX_TIME,
               "num_classes": len(encoder.classes_), "train_size": len(X_train),
               "excluded": sorted(EXCLUDE_CLASSES), "version": "v8",
               "augmentation": True}

start_epoch, best_val_acc, patience_counter = 0, 0.0, 0
train_loss_hist, train_acc_hist = [], []
val_loss_hist,   val_acc_hist   = [], []

if os.path.exists(LATEST):
    try:
        ckpt = torch.load(LATEST, map_location=DEVICE, weights_only=False)
        if ckpt.get("fingerprint") != FINGERPRINT:
            print("\n⚠️  Fingerprint mismatch — starting fresh.")
            os.remove(LATEST)
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_acc = ckpt["best_accuracy"]
            patience_counter = ckpt["patience_counter"]
            train_loss_hist = ckpt["train_loss"]
            train_acc_hist = ckpt["train_accuracy"]
            val_loss_hist = ckpt["val_loss"]
            val_acc_hist = ckpt["val_accuracy"]
            print(f"✅ Resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"Could not resume ({e}) — starting fresh.")
        if os.path.exists(LATEST):
            os.remove(LATEST)

# ============================================================
# TRAINING
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING (v8 — with augmentation)")
print("=" * 70)

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    model.eval()
    val_running, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            val_running += loss.item()
            _, pred = torch.max(out, 1)
            val_total += yb.size(0)
            val_correct += (pred == yb).sum().item()
    val_loss = val_running / len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    scheduler.step(val_loss)

    train_loss_hist.append(train_loss); train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss);     val_acc_hist.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), BEST)
        print(f"  ⭐ New best! Val Acc: {best_val_acc:.2f}%")
    else:
        patience_counter += 1

    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_accuracy": best_val_acc, "patience_counter": patience_counter,
        "train_loss": train_loss_hist, "train_accuracy": train_acc_hist,
        "val_loss": val_loss_hist, "val_accuracy": val_acc_hist,
        "fingerprint": FINGERPRINT,
    }, LATEST)

    print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} | "
          f"TrL: {train_loss:.4f} TrA: {train_acc:>6.2f}% | "
          f"VlL: {val_loss:.4f} VlA: {val_acc:>6.2f}% | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break

print(f"\n✅ Best Val Acc: {best_val_acc:.2f}%")

# ============================================================
# TEST
# ============================================================
print("\n" + "=" * 70)
print(f"TEST ON UNSEEN SUBJECT: {TEST_SUBJECT}")
print("=" * 70)

model.load_state_dict(torch.load(BEST, map_location=DEVICE, weights_only=False))
model.eval()

test_correct, test_total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        _, pred = torch.max(out, 1)
        test_total += yb.size(0)
        test_correct += (pred == yb).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

test_acc = 100.0 * test_correct / test_total
print(f"\n🎯 Test Accuracy: {test_acc:.2f}%  ({test_correct}/{test_total})")

print("\n📋 Per-class report:")
print(classification_report(all_labels, all_preds,
                            labels=list(range(len(encoder.classes_))),
                            target_names=list(encoder.classes_),
                            zero_division=0))

cm = confusion_matrix(all_labels, all_preds,
                      labels=list(range(len(encoder.classes_))))

print("Confusion matrix (rows=true, cols=pred):")
print(f"{'':16s}" + "".join(f"{c[:10]:>12s}" for c in encoder.classes_))
for i, cls in enumerate(encoder.classes_):
    row = f"{cls[:16]:<16s}"
    for j in range(len(encoder.classes_)):
        row += f"{cm[i, j]:>12d}"
    print(row)

# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))
cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": "Row-normalized proportion"})
ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
ax.set_ylabel("True", fontsize=13, labelpad=10)
ax.set_title(f"Confusion Matrix — Test on unseen {TEST_SUBJECT}\n"
             f"Test Acc: {test_acc:.2f}%  |  Features: {FEATURES} (v8 + augment)",
             fontsize=15, pad=15)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(HISTORY_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n📊 Confusion matrix saved: {cm_path}")

# ============================================================
# SAVE
# ============================================================
torch.save(model.state_dict(), FINAL)
torch.save(model, FULL)
print(f"\nFull model: {FULL}")

pd.DataFrame({
    "Epoch": list(range(1, len(train_loss_hist) + 1)),
    "Train Loss": train_loss_hist,
    "Validation Loss": val_loss_hist,
    "Train Accuracy": train_acc_hist,
    "Validation Accuracy": val_acc_hist,
}).to_csv(os.path.join(HISTORY_DIR, "training_history.csv"), index=False)

print("\n" + "=" * 70)
print("🎯 TRAINING COMPLETE — v8 (59 features + augmentation)")
print("=" * 70)
print(f"  Best Val Acc:  {best_val_acc:.2f}%")
print(f"  Test Acc:      {test_acc:.2f}%  (on unseen {TEST_SUBJECT})")
print(f"  Features:      {FEATURES}")
print(f"  Classes:       {len(encoder.classes_)}")
print("=" * 70)