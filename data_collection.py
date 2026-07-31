import cv2
import mediapipe as mp
import json
import os
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_FRAMES = 15
CLASSES       = [("swipe_right", 'r'), ("swipe_left", 'l'), ("swipe_up", 'u'), ("swipe_down", 'd'), ("tap", 't'), ("null", ' ')]
SAVE_PATH     = "data"

# ── Setup ─────────────────────────────────────────────────────────────────────
for c in CLASSES:
    os.makedirs(f"{SAVE_PATH}/{c[0]}", exist_ok=True)

sample_counts = {c[0]: len(os.listdir(f"{SAVE_PATH}/{c[0]}")) for c in CLASSES}
print(sample_counts)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Current label being collected — always "null" unless interrupted
current_label  = "null"
current_buffer = []

# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w    = frame.shape[:2]

        # ── Landmarks ────────────────────────────────────────────────────────
        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hl.landmark]
            current_buffer.append(landmarks)
        else:
            # No hand — if recording null just reset, if mid-gesture also reset
            current_buffer = []

        # ── Save when buffer full ─────────────────────────────────────────────
        if len(current_buffer) == TARGET_FRAMES:
            path = f"{SAVE_PATH}/{current_label}/sample_{sample_counts[current_label]:03d}.json"
            json.dump({"gesture": current_label, "frames": current_buffer}, open(path, "w"))
            sample_counts[current_label] += 1
            print(f"  [{current_label}] saved ({sample_counts[current_label]} total)")

            # Always fall back to null after any save
            current_label  = "null"
            current_buffer = []

        # ── UI ────────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)

        bar_color = {"swipe_right": (100, 255, 100),
                     "swipe_left":  (100, 100, 255),
                     "swipe_up": (255, 255, 0),
                     "swipe_down": (0,0,0),
                     "tap": (255, 255, 255),
                     "null":        (80,  80,  80 )}[current_label]

        # Progress bar
        progress = len(current_buffer) / TARGET_FRAMES
        cv2.rectangle(frame, (0, 55), (int(w * progress), 60), bar_color, -1)

        # Status text
        label_text = f"{current_label.upper()}  {len(current_buffer)}/{TARGET_FRAMES}"
        cv2.putText(frame, label_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)

        # Sample counts
        cv2.rectangle(frame, (0, h - 32), (w, h), (15, 15, 15), -1)
        counts_str = "   ".join(f"{c[0]}: {sample_counts[c[0]]}" for c in CLASSES)
        cv2.putText(frame, counts_str, (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Collect Gestures", frame)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        for motion in CLASSES:
            if key == ord(motion[1]):
                current_label = motion[0]
                current_buffer = []
                print(f"Detected motion - {motion[0]}")
                break
        
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal counts: { {c: sample_counts[c[0]] for c in CLASSES} }")