"""
Dynamic ISL Collection — Class-Foldered Storage + Auto-Advance

Stores per-take CSVs at:
    features/Train_CSV/<CLASS>/<SUBJECT>_<CLASS>_<take>.csv   (SRII, SANJAY)
    features/TEST  CSV/<CLASS>/<SUBJECT>_<CLASS>_<take>.csv   (NANDY)

Behavior:
    - SPACE is pressed ONCE per class
    - After that, all remaining takes for the class auto-advance
    - Auto-skips completed classes and completed takes
    - Press Q anytime to quit and resume later

Usage:
    python collect_dynamic.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import csv
import time
import sys
import shutil

# ============================================================
# CONFIG
# ============================================================
SUBJECT = "SRII"                  # "SRII", "SANJAY", or "NANDY"

WORDS = [
    "HI", "HELLO", "GOOD_MORNING", "GOOD_AFTERNOON", "GOOD_EVENING",
    "THANK_YOU", "PLEASE", "SORRY", "WELCOME", "WHAT",
]

TAKES_MAP = {
    "SRII":   18,
    "SANJAY": 17,
    "NANDY":  15,
}
TAKES_PER_WORD = TAKES_MAP[SUBJECT]

START_FROM_TAKE = 1        # 1 = always start from take 1

FPS = 30
CLIP_DURATION = 2.0
TOTAL_FRAMES = int(FPS * CLIP_DURATION)

COUNTDOWN = 3              # seconds between takes (repositioning time)
SAVE_VIDEO = True
SAVE_FRAMES = True         # save raw .jpg frames per clip
SAVE_MP_IMAGES = True      # save MediaPipe landmark overlays per clip

# Auto-skip completed classes/takes
AUTO_SKIP_DONE_CLASSES = True

# ============================================================
# PATHS (rooted at script location, not CWD)
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR  = os.path.join(ROOT, "videos")
FRAMES_DIR = os.path.join(ROOT, "dataset_images")
MP_DIR     = os.path.join(ROOT, "Mediapipe_images")
META_PATH  = os.path.join(ROOT, "metadata.csv")

FEATURES_ROOT = os.path.join(ROOT, "features")
if SUBJECT == "NANDY":
    FEAT_BASE = os.path.join(FEATURES_ROOT, "TEST CSV")
else:
    FEAT_BASE = os.path.join(FEATURES_ROOT, "Train_CSV")

for d in [VIDEO_DIR, FRAMES_DIR, MP_DIR, FEAT_BASE]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# FEATURE CONFIG (252 features per frame — unchanged)
# ============================================================
POSE_UPPER = list(range(17))
FACE_KEYS = [107, 66, 336, 296, 13, 0, 61, 14, 17, 291]

ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # Pinky
    (2, 0, 5), (5, 0, 9), (13, 0, 17),        # Inter-finger
]

# ============================================================
# HELPERS
# ============================================================
def angle(a, b, c):
    ba, bc = a - b, c - b
    d = np.linalg.norm(ba) * np.linalg.norm(bc)
    if d < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / d, -1, 1))))


def palm_vec(hand):
    wrist, idx, pky = hand[0], hand[5], hand[17]
    n = np.cross(idx - wrist, pky - wrist)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)


def angles(hand):
    return [angle(hand[a], hand[b], hand[c]) for a, b, c in ANGLE_TRIPLETS]


def dist(a, b):
    return float(np.linalg.norm(a - b))


# ============================================================
# FEATURE EXTRACTION (252 features)
# ============================================================
def extract_features(results, prev_lw=None, prev_rw=None):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.left_hand_landmarks.landmark],
                        dtype=np.float32)
    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z]
                          for lm in results.right_hand_landmarks.landmark],
                         dtype=np.float32)

    pose = np.zeros((17, 3), dtype=np.float32)
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x,
                          results.pose_landmarks.landmark[i].y,
                          results.pose_landmarks.landmark[i].z]
                         for i in POSE_UPPER], dtype=np.float32)

    face = np.zeros((10, 3), dtype=np.float32)
    if results.face_landmarks:
        for k, idx in enumerate(FACE_KEYS):
            lm = results.face_landmarks.landmark[idx]
            face[k] = [lm.x, lm.y, lm.z]

    h2h = dist(left[0], right[0]) if (np.any(left) and np.any(right)) else 0.0
    l_vel = dist(left[0], prev_lw) if (prev_lw is not None and np.any(left)) else 0.0
    r_vel = dist(right[0], prev_rw) if (prev_rw is not None and np.any(right)) else 0.0

    feat = np.concatenate([
        left.flatten(), right.flatten(),
        angles(left), angles(right),
        palm_vec(left), palm_vec(right),
        pose.flatten(), face.flatten(),
        [h2h, l_vel, r_vel],
    ]).astype(np.float32)

    assert feat.shape[0] == 252, f"Got {feat.shape[0]}"
    return feat, left[0].copy(), right[0].copy()


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LANDMARK_STYLE   = mp_draw.DrawingSpec(color=(0, 255, 0),     thickness=2, circle_radius=3)
CONNECTION_STYLE = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)


def draw_lms(frame, results):
    out = frame.copy()
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            out, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE)
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            out, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            out, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE)
    return out


# ============================================================
# OVERLAY
# ============================================================
def top_bar(frame, word, take, subject):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)
    cv2.putText(frame, f"Subject: {subject}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Sign: {word}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.putText(frame, f"Take {take}/{TAKES_PER_WORD}", (w - 260, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"{TOTAL_FRAMES} frames @ {FPS} FPS", (w - 300, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return frame


def center_text(frame, text, scale=2, color=(0, 255, 255), thick=4):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(frame, text, ((w - tw)//2, (h + th)//2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


# ============================================================
# PATH RESOLUTION (class-foldered)
# ============================================================
def class_feat_dir(word):
    d = os.path.join(FEAT_BASE, word)
    os.makedirs(d, exist_ok=True)
    return d


def take_paths(clip_name, word):
    cls_dir = class_feat_dir(word)
    return {
        "video":  os.path.join(VIDEO_DIR, f"{clip_name}.mp4"),
        "csv":    os.path.join(cls_dir, f"{clip_name}.csv"),
        "frames": os.path.join(FRAMES_DIR, clip_name),
        "mp":     os.path.join(MP_DIR, clip_name),
    }


def is_take_complete(clip_name, word):
    paths = take_paths(clip_name, word)
    if not os.path.exists(paths["csv"]):
        return False
    try:
        return len(pd.read_csv(paths["csv"])) == TOTAL_FRAMES
    except Exception:
        return False


def is_class_complete(word):
    for take in range(1, TAKES_PER_WORD + 1):
        clip_name = f"{SUBJECT}_{word}_{take-1:02d}"
        if not is_take_complete(clip_name, word):
            return False
    return True


def delete_take(clip_name, word):
    paths = take_paths(clip_name, word)
    if os.path.exists(paths["video"]):
        os.remove(paths["video"])
    if os.path.exists(paths["csv"]):
        os.remove(paths["csv"])
    if os.path.isdir(paths["frames"]):
        shutil.rmtree(paths["frames"])
    if os.path.isdir(paths["mp"]):
        shutil.rmtree(paths["mp"])
    print(f"  Deleted: {clip_name}")


def clean_incomplete_takes():
    print("\n🔍 Scanning for incomplete takes...")
    cleaned = 0
    for word in WORDS:
        for take in range(1, TAKES_PER_WORD + 1):
            clip_name = f"{SUBJECT}_{word}_{take-1:02d}"
            paths = take_paths(clip_name, word)
            any_exists = any([
                os.path.exists(paths["video"]),
                os.path.exists(paths["csv"]),
                os.path.isdir(paths["frames"]),
                os.path.isdir(paths["mp"]),
            ])
            if any_exists and not is_take_complete(clip_name, word):
                delete_take(clip_name, word)
                cleaned += 1
    print(f"🧹 Cleaned {cleaned} incomplete takes\n")


# ============================================================
# METADATA
# ============================================================
if not os.path.exists(META_PATH):
    with open(META_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["subject", "word", "take", "num_frames",
                                 "video", "csv", "frames_dir", "mp_dir"])

# ============================================================
# WEBCAM
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, FPS)
if not cap.isOpened():
    raise Exception("Cannot open webcam")

# ============================================================
# STARTUP BANNER + CLASS STATUS
# ============================================================
print("=" * 70)
print(f"DYNAMIC ISL COLLECTION — Subject: {SUBJECT}")
print("=" * 70)
print(f"Project root:    {ROOT}")
print(f"Features base:   {FEAT_BASE}")
print(f"Words:           {len(WORDS)}")
print(f"Takes per word:  {TAKES_PER_WORD}")
print(f"Duration/clip:   {CLIP_DURATION}s ({TOTAL_FRAMES} frames)")
print(f"Features/frame:  252")
print()

print("📊 Class status:")
for word in WORDS:
    n_done = sum(1 for t in range(1, TAKES_PER_WORD + 1)
                 if is_take_complete(f"{SUBJECT}_{word}_{t-1:02d}", word))
    status = "✅ DONE" if n_done == TAKES_PER_WORD else f"⏳ {n_done}/{TAKES_PER_WORD}"
    print(f"   {word:<18s}  {status}")

print()
print("Controls:")
print("  SPACE → start recording this class (auto-advances through all takes)")
print("  Q     → quit session (can resume later)")
print("=" * 70)

# Clean incomplete takes before starting
clean_incomplete_takes()

skipped = 0
completed = 0

# ============================================================
# MAIN LOOP  (SPACE once per class → auto-advance takes)
# ============================================================
for word in WORDS:
    if AUTO_SKIP_DONE_CLASSES and is_class_complete(word):
        print(f"\n{'='*70}")
        print(f"⏭️  SKIPPING '{word}' — already complete")
        print(f"{'='*70}")
        continue

    print(f"\n{'='*70}\nSIGN: {word}\n{'='*70}")

    # ---- IDLE (once per class) ----
    start = False
    while not start:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = top_bar(frame, word, "—", SUBJECT)
        center_text(frame, "PRESS SPACE TO START", 1.4)
        cv2.imshow("Dynamic ISL Collection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:               # SPACE
            start = True
        elif k == ord('q'):
            cap.release(); holistic.close(); cv2.destroyAllWindows()
            print("\nQuit by user.")
            sys.exit(0)

    # ---- AUTO-RECORD ALL REMAINING TAKES FOR THIS CLASS ----
    for take in range(START_FROM_TAKE, TAKES_PER_WORD + 1):
        clip_name = f"{SUBJECT}_{word}_{take-1:02d}"
        paths = take_paths(clip_name, word)

        if is_take_complete(clip_name, word):
            print(f"  Take {take}: complete, skipping")
            continue

        # ---- COUNTDOWN ----
        for c in range(COUNTDOWN, 0, -1):
            t0 = time.time()
            while time.time() - t0 < 1.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame = top_bar(frame, word, take, SUBJECT)
                center_text(frame, str(c), 5, (0, 0, 255), 10)
                cv2.imshow("Dynamic ISL Collection", frame)
                cv2.waitKey(1)

        # ---- RECORD ----
        print(f"  Take {take}: recording...", end=" ", flush=True)
        frames_raw = []
        frames_mp = []
        features_all = []
        prev_lw = None
        prev_rw = None
        t0 = time.time()

        for fi in range(TOTAL_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            feat, lw, rw = extract_features(results, prev_lw, prev_rw)
            features_all.append(feat)
            if np.any(lw):
                prev_lw = lw
            if np.any(rw):
                prev_rw = rw

            frames_raw.append(frame.copy())
            if SAVE_MP_IMAGES:
                frames_mp.append(draw_lms(frame, results))

            prev = draw_lms(frame, results) if SAVE_MP_IMAGES else frame.copy()
            prev = top_bar(prev, word, take, SUBJECT)
            cv2.circle(prev, (40, 130), 15, (0, 0, 255), -1)
            cv2.putText(prev, f"REC {fi+1}/{TOTAL_FRAMES}",
                        (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Dynamic ISL Collection", prev)

            elapsed = time.time() - t0
            expected = (fi + 1) / FPS
            if elapsed < expected:
                time.sleep(expected - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ---- SAVE ----
        if len(frames_raw) == TOTAL_FRAMES and len(features_all) == TOTAL_FRAMES:
            # Video
            if SAVE_VIDEO:
                h, w = frames_raw[0].shape[:2]
                vw = cv2.VideoWriter(paths["video"],
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     FPS, (w, h))
                for fr in frames_raw:
                    vw.write(fr)
                vw.release()

            # Raw frames
            if SAVE_FRAMES:
                os.makedirs(paths["frames"], exist_ok=True)
                for i, fr in enumerate(frames_raw):
                    cv2.imwrite(os.path.join(paths["frames"],
                                             f"{clip_name}_{i:03d}.jpg"), fr)

            # MediaPipe frames
            if SAVE_MP_IMAGES:
                os.makedirs(paths["mp"], exist_ok=True)
                for i, fr in enumerate(frames_mp):
                    cv2.imwrite(os.path.join(paths["mp"],
                                             f"{clip_name}_{i:03d}.jpg"), fr)

            # CSV (with 'frame' column, 252 features)
            arr = np.array(features_all, dtype=np.float32)
            df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(252)])
            df.insert(0, "frame", np.arange(len(arr)))
            df.to_csv(paths["csv"], index=False)

            with open(META_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    SUBJECT, word, take-1, len(arr),
                    paths["video"], paths["csv"], paths["frames"], paths["mp"]
                ])
            print(f"saved → {paths['csv']}")
            completed += 1
        else:
            print(f"INCOMPLETE ({len(frames_raw)}/{TOTAL_FRAMES} frames) — deleting")
            delete_take(clip_name, word)
            skipped += 1

        time.sleep(0.4)

cap.release()
holistic.close()
cv2.destroyAllWindows()

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print(f"SESSION COMPLETE — Subject: {SUBJECT}")
print(f"  Completed takes this run: {completed}")
print(f"  Skipped takes this run:   {skipped}")
print("=" * 70)

print("\n📊 Final class status:")
for word in WORDS:
    n_done = sum(1 for t in range(1, TAKES_PER_WORD + 1)
                 if is_take_complete(f"{SUBJECT}_{word}_{t-1:02d}", word))
    status = "✅ DONE" if n_done == TAKES_PER_WORD else f"⏳ {n_done}/{TAKES_PER_WORD}"
    print(f"   {word:<18s}  {status}")
print("=" * 70)