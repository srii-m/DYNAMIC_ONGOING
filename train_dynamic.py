"""
Dynamic ISL Training — Cross-Person Evaluation with New Folder Structure

Data flow (STRICT):
    features/Train_CSV/<CLASS>/{SRII, SANJAY}_*.csv   →  Training + Validation
    features/TEST  CSV/<CLASS>/NANDY_*.csv            →  Test ONLY (unseen person)

Pipeline:
    - Recursively discover CSVs in class subfolders
    - Uniform temporal resampling to MAX_TIME = 60
    - StandardScaler fit on TRAIN only
    - Train UnifiedISLNet with class weights + early stopping
    - Evaluate on NANDY test set
    - Export PyTorch / TorchScript / ONNX models

Author: (your team)
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# IMPORT MODEL (relative to this script's location)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from dynamic_cnn import UnifiedISLNet

warnings.filterwarnings("ignore")

# ============================================================
# PATHS  (all relative to SCRIPT_DIR, not CWD)
# ============================================================
FEATURES_ROOT   = os.path.join(SCRIPT_DIR, "features")
TRAIN_DIR       = os.path.join(FEATURES_ROOT, "Train_CSV")
TEST_DIR        = os.path.join(FEATURES_ROOT, "TEST CSV")

OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "outputs_dynamic")
CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_DIR      = os.path.join(OUTPUT_DIR, "history")
DATA_SPLIT_DIR   = os.path.join(OUTPUT_DIR, "data_splits")
MODEL_EXPORT_DIR = os.path.join(SCRIPT_DIR, "models_dynamic")

for d in [CHECKPOINT_DIR, HISTORY_DIR, DATA_SPLIT_DIR, MODEL_EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MAX_TIME  = 60
FEATURES  = 252
BATCH_SIZE = 32
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42

TRAIN_SUBJECTS = {"SRII", "SANJAY"}
TEST_SUBJECT   = "NANDY"

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
print(f"DEVICE: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

# ============================================================
# TEMPORAL RESAMPLING
# ============================================================
def resample_clip(clip, target_len=MAX_TIME):
    """
    Uniform temporal resampling.
      - T == target: return unchanged
      - T >  target: sample target evenly-spaced indices (preserves start & end)
      - T <  target: zero-pad at the end
    """
    T, F = clip.shape
    if T == target_len:
        return clip
    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        return clip[indices]
    pad = np.zeros((target_len - T, F), dtype=clip.dtype)
    return np.vstack([clip, pad])


# ============================================================
# CSV DISCOVERY (folder name = label, filename = subject_id)
# ============================================================
def discover_csvs(root_dir):
    """
    Walk <root_dir>/<CLASS>/<subject>_<class>_<id>.csv
    Returns list of (csv_path, subject, label, class_folder).
    """
    entries = []
    if not os.path.isdir(root_dir):
        print(f"⚠️  Folder not found: {root_dir}")
        return entries

    for class_dir in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        label = class_dir  # folder name IS the label
        for csv_path in sorted(glob.glob(os.path.join(class_path, "*.csv"))):
            base = os.path.basename(csv_path).replace(".csv", "")
            subject = base.split("_")[0]
            entries.append((csv_path, subject, label, class_dir))
    return entries


def load_clips(entries, split_name):
    """Load all CSVs in `entries`, resample, return arrays + metadata."""
    clips, labels, subjects = [], [], []
    stats = {"unchanged": 0, "resampled": 0, "padded": 0}
    for csv_path, subject, label, _ in entries:
        df = pd.read_csv(csv_path)

        # Drop the 'frame' index column if present (it's not a real feature)
        if "frame" in df.columns:
            df = df.drop(columns=["frame"])

        # Also drop any unnamed index columns just in case
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        clip = df.values.astype(np.float32)
        T = clip.shape[0]

        clip = resample_clip(clip, MAX_TIME)

        if T == MAX_TIME:
            stats["unchanged"] += 1
        elif T > MAX_TIME:
            stats["resampled"] += 1
        else:
            stats["padded"] += 1

        clips.append(clip)
        labels.append(label)
        subjects.append(subject)

    if not clips:
        print(f"❌ No clips found for {split_name}.")
        return None, None, None, stats

    X = np.stack(clips, axis=0)                # (N, 60, 252)
    y = np.array(labels)
    s = np.array(subjects)

    print(f"\n[{split_name}] Loaded {len(clips)} clips")
    print(f"  Unchanged (T={MAX_TIME}): {stats['unchanged']}")
    print(f"  Resampled (T>{MAX_TIME}): {stats['resampled']}")
    print(f"  Padded    (T<{MAX_TIME}): {stats['padded']}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes ({len(set(y))}): {sorted(set(y))}")
    print(f"  Subjects: {sorted(set(s))}")
    return X, y, s, stats


# ============================================================
# DISCOVER FILES
# ============================================================
print("\n" + "=" * 70)
print("DISCOVERING DATASET")
print("=" * 70)
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TEST_DIR:  {TEST_DIR}")

train_entries = discover_csvs(TRAIN_DIR)
test_entries  = discover_csvs(TEST_DIR)

# ============================================================
# LOAD DATA
# ============================================================
X_train_pool, y_train_pool, s_train_pool, _ = load_clips(train_entries, "TRAIN/VAL POOL")
X_test,       y_test,       s_test,       _ = load_clips(test_entries,  "TEST")

if X_train_pool is None or X_test is None:
    print("\n❌ Cannot proceed without both training and test data.")
    sys.exit(1)

# ============================================================
# DATASET SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nTRAIN/VALIDATION DATA:")
print(f"  Total clips: {len(X_train_pool)}")
for subj in sorted(set(s_train_pool)):
    n = int((s_train_pool == subj).sum())
    print(f"  {subj} clips: {n}")
print(f"  Classes: {sorted(set(y_train_pool))}")

print("\nTEST DATA:")
print(f"  Total {TEST_SUBJECT} clips: {len(X_test)}")
print(f"  Classes: {sorted(set(y_test))}")

# Sanity: subject separation
assert set(s_train_pool).issubset(TRAIN_SUBJECTS), \
    f"Unexpected subjects in train pool: {set(s_train_pool) - TRAIN_SUBJECTS}"
assert set(s_test).issubset({TEST_SUBJECT}), \
    f"Unexpected subjects in test: {set(s_test) - {TEST_SUBJECT}}"
assert len(set(s_train_pool) & set(s_test)) == 0, \
    "❌ SUBJECT LEAK: overlap between train and test subjects!"

print("\n✅ No subject leakage: train/test subjects are disjoint.")

# ============================================================
# TRAIN / VALIDATION SPLIT (from TRAIN POOL only)
# ============================================================
idx = np.arange(len(X_train_pool))
tr_idx, va_idx = train_test_split(
    idx, test_size=0.15, random_state=SEED, stratify=y_train_pool
)

X_train, y_train = X_train_pool[tr_idx], y_train_pool[tr_idx]
X_val,   y_val   = X_train_pool[va_idx], y_train_pool[va_idx]

print("\n" + "=" * 70)
print("FINAL SPLIT SIZES")
print("=" * 70)
print(f"TRAIN:      X={X_train.shape}   y={y_train.shape}")
print(f"VALIDATION: X={X_val.shape}     y={y_val.shape}")
print(f"TEST:       X={X_test.shape}    y={y_test.shape}  (subject: {TEST_SUBJECT})")

# ============================================================
# LABEL ENCODING (fit on train only)
# ============================================================
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc   = encoder.transform(y_val)
y_test_enc  = encoder.transform(y_test)
joblib.dump(encoder, os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"))
print(f"\nLabel encoder saved ({len(encoder.classes_)} classes): {list(encoder.classes_)}")

# ============================================================
# FEATURE SCALING (fit on train only)
# ============================================================
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, FEATURES))

X_train_s = scaler.transform(X_train.reshape(-1, FEATURES)).reshape(-1, MAX_TIME, FEATURES)
X_val_s   = scaler.transform(X_val.reshape(-1, FEATURES)).reshape(-1, MAX_TIME, FEATURES)
X_test_s  = scaler.transform(X_test.reshape(-1, FEATURES)).reshape(-1, MAX_TIME, FEATURES)

joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
print("Scaler fit on TRAIN only, applied to val/test.")

# Save splits
np.save(os.path.join(DATA_SPLIT_DIR, "X_train.npy"), X_train_s)
np.save(os.path.join(DATA_SPLIT_DIR, "y_train.npy"), y_train_enc)
np.save(os.path.join(DATA_SPLIT_DIR, "X_val.npy"),   X_val_s)
np.save(os.path.join(DATA_SPLIT_DIR, "y_val.npy"),   y_val_enc)
np.save(os.path.join(DATA_SPLIT_DIR, "X_test.npy"),  X_test_s)
np.save(os.path.join(DATA_SPLIT_DIR, "y_test.npy"),  y_test_enc)
print(f"Data splits saved to: {DATA_SPLIT_DIR}/")

# ============================================================
# TORCH TENSORS + LOADERS
# ============================================================
X_train_t = torch.tensor(X_train_s, dtype=torch.float32).unsqueeze(1)  # (N,1,60,252)
X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32).unsqueeze(1)

y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
y_val_t   = torch.tensor(y_val_enc,   dtype=torch.long)
y_test_t  = torch.tensor(y_test_enc,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                          batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL
# ============================================================
model = UnifiedISLNet(
    num_classes=len(encoder.classes_),
    max_time=MAX_TIME,
    features=FEATURES,
).to(DEVICE)
print(f"\nModel: {model.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# LOSS + OPTIMIZER + SCHEDULER
# ============================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc,
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ============================================================
# CHECKPOINT PATHS
# ============================================================
LATEST = os.path.join(CHECKPOINT_DIR, "latest.pth")
BEST   = os.path.join(CHECKPOINT_DIR, "best_model.pth")
FINAL  = os.path.join(CHECKPOINT_DIR, "final_model.pth")
FULL   = os.path.join(MODEL_EXPORT_DIR, "full_model.pth")

# ============================================================
# RESUME SUPPORT
# ============================================================
start_epoch, best_val_acc, patience_counter = 0, 0.0, 0
train_loss_hist, train_acc_hist = [], []
val_loss_hist, val_acc_hist = [], []

if os.path.exists(LATEST):
    try:
        ckpt = torch.load(LATEST, map_location=DEVICE, weights_only=False)
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
        print(f"Resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"⚠️  Could not resume ({e}), starting fresh.")
        if os.path.exists(LATEST):
            os.remove(LATEST)

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(start_epoch, NUM_EPOCHS):
    # ---------- Train ----------
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

    # ---------- Validate ----------
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

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    # ---------- Checkpoint ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), BEST)
        print(f"  ⭐ New best! Val Acc: {best_val_acc:.2f}%")
    else:
        patience_counter += 1

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_accuracy": best_val_acc,
        "patience_counter": patience_counter,
        "train_loss": train_loss_hist,
        "train_accuracy": train_acc_hist,
        "val_loss": val_loss_hist,
        "val_accuracy": val_acc_hist,
    }, LATEST)

    print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:>6.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:>6.2f}% | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered!")
        break

print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")

# ============================================================
# TEST ON UNSEEN SUBJECT
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
print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
print(f"   Correct: {test_correct}/{test_total}")

# Per-class report
from sklearn.metrics import classification_report, confusion_matrix
print("\n📋 Per-class report:")
print(classification_report(
    all_labels, all_preds,
    labels=list(range(len(encoder.classes_))),
    target_names=list(encoder.classes_),
    zero_division=0,
))

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
# SAVE MODELS
# ============================================================
print("\nSaving models...")
torch.save(model.state_dict(), FINAL)
torch.save(model, FULL)
print(f"  Full model (.pth): {FULL}")

# TorchScript
try:
    model.eval()
    dummy = torch.randn(1, 1, MAX_TIME, FEATURES).to(DEVICE)
    scripted = torch.jit.trace(model, dummy)
    ts_path = os.path.join(MODEL_EXPORT_DIR, "model_scripted.pt")
    scripted.save(ts_path)
    print(f"  TorchScript (.pt): {ts_path}")
except Exception as e:
    print(f"  TorchScript failed: {e}")

# ONNX
try:
    model_cpu = model.to("cpu").eval()
    dummy_cpu = torch.randn(1, 1, MAX_TIME, FEATURES)
    onnx_path = os.path.join(MODEL_EXPORT_DIR, "model.onnx")
    torch.onnx.export(
        model_cpu, dummy_cpu, onnx_path,
        export_params=True, opset_version=13,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"  ONNX (.onnx): {onnx_path}")
    model.to(DEVICE)
except Exception as e:
    print(f"  ONNX failed: {e}")

# History
pd.DataFrame({
    "Epoch": list(range(1, len(train_loss_hist) + 1)),
    "Train Loss": train_loss_hist,
    "Validation Loss": val_loss_hist,
    "Train Accuracy": train_acc_hist,
    "Validation Accuracy": val_acc_hist,
}).to_csv(os.path.join(HISTORY_DIR, "training_history.csv"), index=False)
print(f"  History: {HISTORY_DIR}/training_history.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("🎯 TRAINING COMPLETE")
print("=" * 70)
print(f"  Best Val Acc:  {best_val_acc:.2f}%")
print(f"  Test Acc:      {test_acc:.2f}%  (on unseen {TEST_SUBJECT})")
print(f"  Model:         {FULL}")
print(f"  Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
print("=" * 70)