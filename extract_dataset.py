"""
Dynamic ISL Feature Extraction — Named Columns Version
Saves per-clip CSVs with descriptive column names (not f0, f1, ...).
"""

import os
import sys
import csv
import glob
import cv2
import numpy as np
import mediapipe as mp
from tkinter import Tk, filedialog

# ============================================================
# CONFIG
# ============================================================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_GLOBAL_CSV = os.path.join(PROJECT_PATH, "dataset", "features", "features_252.csv")
CLIP_DIR = os.path.join(PROJECT_PATH, "dataset_features")

os.makedirs(os.path.dirname(OUTPUT_GLOBAL_CSV), exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ============================================================
# FEATURE CONFIG
# ============================================================
POSE_UPPER = list(range(17))
FACE_KEYS  = [107, 66, 336, 296, 13, 0, 61, 14, 17, 291]

ANGLE_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),
    (0,5,6),(5,6,7),(6,7,8),
    (0,9,10),(9,10,11),(10,11,12),
    (0,13,14),(13,14,15),(14,15,16),
    (0,17,18),(17,18,19),(18,19,20),
    (2,0,5),(5,0,9),(13,0,17),
]

# ============================================================
# HELPERS
# ============================================================
def calculate_angle(a, b, c):
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def palm_orientation(hand):
    w, i, p = hand[0], hand[5], hand[17]
    normal = np.cross(i - w, p - w)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0], np.float32)
    return normal / norm

def extract_angles(hand):
    return [calculate_angle(hand[a], hand[b], hand[c]) for a,b,c in ANGLE_TRIPLETS]


def extract_features(results):
    """Return 252-float vector for one frame."""
    lh = np.zeros((21, 3), np.float32)
    rh = np.zeros((21, 3), np.float32)
    pu = np.zeros((17, 3), np.float32)
    fk = np.zeros((10, 3), np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark], np.float32)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark], np.float32)
    if results.pose_landmarks:
        pu = np.array([[results.pose_landmarks.landmark[i].x,
                        results.pose_landmarks.landmark[i].y,
                        results.pose_landmarks.landmark[i].z]
                       for i in POSE_UPPER], np.float32)
    if results.face_landmarks:
        fk = np.array([[results.face_landmarks.landmark[i].x,
                        results.face_landmarks.landmark[i].y,
                        results.face_landmarks.landmark[i].z]
                       for i in FACE_KEYS], np.float32)

    feats = []
    feats.extend(lh.flatten().tolist())              # 63
    feats.extend(rh.flatten().tolist())              # 63
    feats.extend(extract_angles(lh))                 # 18
    feats.extend(extract_angles(rh))                 # 18
    feats.extend(palm_orientation(lh).tolist())      # 3
    feats.extend(palm_orientation(rh).tolist())      # 3
    feats.extend(pu.flatten().tolist())              # 51
    feats.extend(fk.flatten().tolist())              # 30
    feats.append(float(np.linalg.norm(lh[0] - rh[0])))  # 1
    feats.append(0.0)                                    # 1 velocity
    feats.append(0.0)                                    # 1 velocity
    assert len(feats) == 252
    return np.array(feats, np.float32)


def compute_velocity(arr):
    T = arr.shape[0]
    for t in range(1, T):
        lw_now, lw_prev = arr[t, 0:3], arr[t-1, 0:3]
        rw_now, rw_prev = arr[t, 63:66], arr[t-1, 63:66]
        arr[t, 250] = float(np.linalg.norm(lw_now - lw_prev))
        arr[t, 251] = float(np.linalg.norm(rw_now - rw_prev))
    return arr


# ============================================================
# ⭐ BUILD NAMED HEADER (252 columns + frame_idx)
# ============================================================
def build_named_header():
    """
    Returns list of column names for the per-clip CSV:
    [frame_idx, L_x0, L_y0, L_z0, ..., R_wrist_vel]
    Total: 1 + 252 = 253 columns.
    """
    cols = ["frame_idx"]

    # 1) Left hand landmarks (63)
    for i in range(21):
        cols.extend([f"L_x{i}", f"L_y{i}", f"L_z{i}"])

    # 2) Right hand landmarks (63)
    for i in range(21):
        cols.extend([f"R_x{i}", f"R_y{i}", f"R_z{i}"])

    # 3) Left hand angles (18)
    for i in range(1, 19):
        cols.append(f"L_angle_{i}")

    # 4) Right hand angles (18)
    for i in range(1, 19):
        cols.append(f"R_angle_{i}")

    # 5) Palm orientations (6)
    cols.extend(["L_palm_x", "L_palm_y", "L_palm_z"])
    cols.extend(["R_palm_x", "R_palm_y", "R_palm_z"])

    # 6) Upper pose (51)
    for i in range(17):
        cols.extend([f"P_x{i}", f"P_y{i}", f"P_z{i}"])

    # 7) Face keypoints (30)
    for i in range(10):
        cols.extend([f"F_x{i}", f"F_y{i}", f"F_z{i}"])

    # 8) Computed extras (3)
    cols.extend(["hand_distance", "L_wrist_vel", "R_wrist_vel"])

    assert len(cols) == 253, f"Header length: {len(cols)}"
    return cols


# ============================================================
# ⭐ BUILD GLOBAL CSV HEADER (adds label + clip_name)
# ============================================================
def build_global_header():
    """[label, clip_name] + per-clip columns (without frame_idx duplication)"""
    return ["label", "clip_name"] + build_named_header()


# ============================================================
# SELECT FOLDER
# ============================================================
print("=" * 70)
print("DYNAMIC ISL FEATURE EXTRACTION — NAMED COLUMNS")
print("=" * 70)

root = Tk()
root.withdraw()
root.attributes("-topmost", True)
chosen_folder = filedialog.askdirectory(title="Select root folder")
if not chosen_folder:
    print("❌ No folder selected.")
    sys.exit(0)
print(f"✅ Selected: {chosen_folder}\n")

# ============================================================
# DISCOVER STRUCTURE
# ============================================================
def find_clips_in_class(class_path):
    """Returns list of (clip_name, [image_paths]) — handles nested OR flat."""
    entries = os.listdir(class_path)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(class_path, e))]
    imgs_here = []
    for ext in IMAGE_EXTS:
        imgs_here.extend(glob.glob(os.path.join(class_path, f"*{ext}")))
        imgs_here.extend(glob.glob(os.path.join(class_path, f"*{ext.upper()}")))
    imgs_here = sorted(set(imgs_here))

    if imgs_here and not subdirs:
        return [("clip", imgs_here)]

    clips = []
    for sub in subdirs:
        sub_path = os.path.join(class_path, sub)
        imgs = []
        for ext in IMAGE_EXTS:
            imgs.extend(glob.glob(os.path.join(sub_path, f"*{ext}")))
            imgs.extend(glob.glob(os.path.join(sub_path, f"*{ext.upper()}")))
        imgs = sorted(set(imgs))
        if imgs:
            clips.append((sub, imgs))
    return clips


class_folders = sorted([
    f for f in os.listdir(chosen_folder)
    if os.path.isdir(os.path.join(chosen_folder, f))
])

print(f"Found {len(class_folders)} classes:")
for c in class_folders:
    clips = find_clips_in_class(os.path.join(chosen_folder, c))
    print(f"  {c}: {len(clips)} clips")
print()

# ============================================================
# MEDIAPIPE
# ============================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
)

# ============================================================
# MAIN LOOP
# ============================================================
per_clip_header = build_named_header()      # 253 names
global_header   = build_global_header()     # 255 names (label, clip_name + 253)

all_global_rows = [global_header]
total_clips, total_frames, errors = 0, 0, 0

print("=" * 70)
print("PROCESSING")
print("=" * 70)

for class_name in class_folders:
    class_path = os.path.join(chosen_folder, class_name)
    clips = find_clips_in_class(class_path)
    print(f"\n📁 Class: {class_name}  ({len(clips)} clips)")

    for clip_name, image_files in clips:
        frames = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            feat = extract_features(results)
            frames.append(feat)

        if not frames:
            print(f"  ⚠️  {clip_name}: extraction failed")
            errors += 1
            continue

        arr = np.array(frames, dtype=np.float32)   # (T, 252)
        arr = compute_velocity(arr)

        # ---- Per-clip CSV with NAMED columns ----
        clip_csv = os.path.join(CLIP_DIR, f"{class_name}_{clip_name}.csv")
        with open(clip_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(per_clip_header)                       # named header
            for t in range(arr.shape[0]):
                w.writerow([t] + arr[t].tolist())             # frame_idx + 252 features

        # ---- Global CSV ----
        for t in range(arr.shape[0]):
            all_global_rows.append(
                [class_name, clip_name] + [t] + arr[t].tolist()
            )

        print(f"  ✅ {clip_name}: {len(frames)} frames → {class_name}_{clip_name}.csv")
        total_clips += 1
        total_frames += len(frames)

holistic.close()

# Save global CSV with named columns
with open(OUTPUT_GLOBAL_CSV, "w", newline="") as f:
    csv.writer(f).writerows(all_global_rows)

print("\n" + "=" * 70)
print("✅ EXTRACTION COMPLETE")
print("=" * 70)
print(f"  Classes:         {len(class_folders)}")
print(f"  Clips:           {total_clips}")
print(f"  Frames:          {total_frames}")
print(f"  Features/frame:  252")
print(f"  Per-clip header: {len(per_clip_header)} columns")
print(f"  Global header:   {len(global_header)} columns")
print(f"  Errors:          {errors}")
print(f"  Global CSV:      {OUTPUT_GLOBAL_CSV}")
print(f"  Per-clip CSVs:   {CLIP_DIR}/")
print("=" * 70)