"""
Gesture TV remote — LG webOS edition.

Right hand: LSTM + peak detection (swipe_right/left/up/down, tap)
            mapped to TV arrow/ok commands.
Left hand:  finger-count presets -> app launch.
WebOS:      connection + command layer.

Usage:
  python main.py --webos-host 10.0.0.6
  python main.py --webos-host 10.0.0.6 --webos-insecure
  python main.py --webos-host 10.0.0.6 --presets "1:netflix,2:youtube.leanback.v4"
  python main.py --list-apps --webos-host 10.0.0.6
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
import json
import collections
from collections import deque
from pathlib import Path
from typing import Optional
from scipy.signal import find_peaks

# ── Fixed model config (must match training) ──────────────────────────────────
CLASSES       = ["swipe_right", "swipe_left", "null", "swipe_up", "swipe_down", "tap"]
TARGET_FRAMES = 15
HIDDEN        = 128
NUM_LAYERS    = 2
MODEL_PATH    = "gesture_model.pt"

# ── Inference defaults (tunable via args) ─────────────────────────────────────
DEFAULT_CONFIDENCE = {
    "swipe_right": 0.75,
    "swipe_left":  0.75,
    "swipe_up":    0.75,
    "swipe_down":  0.75,
    "tap":         0.9,
}
PROMINENCE    = 0.30
PEAK_DISTANCE = 10
PEAK_LAG      = 3
HISTORY_LEN   = 30
COOLDOWN      = 0.6
MAX_MISSED    = 6

GESTURE_TO_CMD = {
    "swipe_right": "right",
    "swipe_left":  "left",
    "swipe_up":    "up",
    "swipe_down":  "down",
    "tap":         "ok",
}

DEFAULT_PRESETS = {
    1: "netflix",
    2: "youtube.leanback.v4",
    3: "com.webos.app.livetv",
    4: "amazon",
}


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Gesture TV Remote — LG webOS")

    # Connection
    p.add_argument("--webos-host",     default="10.0.0.6",
                   help="LG webOS TV IP address (default: 10.0.0.6)")
    p.add_argument("--webos-store",    default="webos_store.json",
                   help="Pairing token file path (default: webos_store.json)")
    p.add_argument("--webos-insecure", action="store_true",
                   help="Use ws:// instead of wss://")
    p.add_argument("--no-tv",          action="store_true",
                   help="Run without connecting to TV (gesture print-only mode)")

    # App presets
    p.add_argument("--presets", default=None,
                   help="Left-hand presets: comma-separated count:app_id pairs "
                        "(e.g. '1:netflix,2:youtube.leanback.v4'). "
                        "Overrides built-in defaults.")
    p.add_argument("--list-apps", action="store_true",
                   help="Print all installed app IDs and exit")

    # Camera
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # Detection tuning
    p.add_argument("--cooldown",      type=float, default=COOLDOWN,
                   help=f"Seconds between gesture triggers (default: {COOLDOWN})")
    p.add_argument("--prominence",    type=float, default=PROMINENCE,
                   help=f"Peak prominence threshold (default: {PROMINENCE})")
    p.add_argument("--peak-lag",      type=int,   default=PEAK_LAG,
                   help=f"Frames past peak before firing (default: {PEAK_LAG})")
    p.add_argument("--max-missed",    type=int,   default=MAX_MISSED,
                   help=f"No-hand frames before buffer reset (default: {MAX_MISSED})")

    return p.parse_args()


def parse_presets(presets_str: str) -> dict:
    """Parse '1:netflix,2:youtube.leanback.v4' into {1: 'netflix', ...}"""
    result = {}
    for pair in presets_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            count_str, app_id = pair.split(":", 1)
            try:
                result[int(count_str.strip())] = app_id.strip()
            except ValueError:
                print(f"Warning: skipping invalid preset '{pair}'")
    return result


# ── WebOS remote ──────────────────────────────────────────────────────────────
class WebOSRemote:
    def __init__(self, host: str, store_path: str = "webos_store.json",
                 secure: bool = True):
        try:
            from pywebostv.connection import WebOSClient
            from pywebostv.controls import InputControl, MediaControl, SystemControl
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pip install pywebostv") from exc

        self.WebOSClient   = WebOSClient
        self.InputControl  = InputControl
        self.MediaControl  = MediaControl
        self.SystemControl = SystemControl

        self.host            = host
        self.secure          = secure
        self.store_path      = Path(store_path)
        self.store           = self._load_store()
        self.client          = None
        self.input           = None
        self.media           = None
        self.system          = None
        self.app_control     = None
        self.input_connected = False

    def _load_store(self) -> dict:
        if self.store_path.exists():
            try:
                return json.loads(self.store_path.read_text())
            except json.JSONDecodeError:
                print(f"Warning: invalid pairing store at {self.store_path}. Re-pairing.")
        return {}

    def _save_store(self) -> None:
        self.store_path.write_text(json.dumps(self.store, indent=2))

    def connect(self) -> None:
        self.client = self.WebOSClient(self.host, secure=self.secure)
        self.client.connect()
        for status in self.client.register(self.store):
            if status == self.WebOSClient.PROMPTED:
                print("Accept the pairing prompt on the LG TV.")
            elif status == self.WebOSClient.REGISTERED:
                print("Registered with LG webOS TV.")
        self._save_store()

        self.input  = self.InputControl(self.client)
        self.media  = self.MediaControl(self.client)
        self.system = self.SystemControl(self.client)

        try:
            from pywebostv.controls import ApplicationControl
            self.app_control = ApplicationControl(self.client)
        except Exception:
            self.app_control = None

        self._connect_input_socket()

    def _connect_input_socket(self) -> None:
        if self.input_connected:
            return
        self.input.connect_input()
        self.input_connected = True

    def _safe_input(self, method_name: str) -> None:
        try:
            self._connect_input_socket()
            getattr(self.input, method_name)()
        except Exception:
            self.input_connected = False
            try:
                self._connect_input_socket()
                getattr(self.input, method_name)()
            except Exception as exc:
                print(f"[webOS] Input command failed: {exc}")

    def send(self, command: str) -> None:
        if self.client is None:
            self.connect()
        command = command.lower().strip()
        input_map = {
            "left": "left", "right": "right", "up": "up", "down": "down",
            "ok": "ok", "enter": "ok", "back": "back", "home": "home",
            "menu": "menu", "play": "play", "pause": "pause",
            "stop": "stop", "mute": "mute",
        }
        media_map = {
            "volup": "volume_up", "voldown": "volume_down",
            "volume_up": "volume_up", "volume_down": "volume_down",
        }
        if command in input_map:
            self._safe_input(input_map[command])
            print(f"[webOS] {command}")
        elif command in media_map:
            getattr(self.media, media_map[command])()
            print(f"[webOS] {command}")
        elif command == "notify":
            self.system.notify("Gesture remote connected")
        else:
            print(f"[webOS] Unsupported command: {command}")

    def list_apps(self) -> dict:
        """Returns {title: app_id} dict. Used by the GUI launcher."""
        if self.client is None:
            self.connect()
        if self.app_control is None:
            raise RuntimeError("ApplicationControl unavailable")
        apps = self.app_control.list_apps()
        return {a.data.get("title", "?"): a.data.get("id", "") for a in apps}

    def launch_app(self, app_id: str) -> None:
        if self.client is None:
            self.connect()
        if self.app_control is None:
            print(f"[webOS] ApplicationControl unavailable, cannot launch {app_id}")
            return
        try:
            matched = [a for a in self.app_control.list_apps()
                       if a.data.get("id") == app_id]
            app_obj = matched[0] if matched else app_id
            self.app_control.launch(app_obj)
            print(f"[webOS] Launched: {app_id}")
        except Exception as exc:
            print(f"[webOS] Launch failed ({app_id}): {exc}")

    def close(self) -> None:
        if self.input_connected:
            try:
                self.input.disconnect_input()
            except Exception:
                pass
        self.input_connected = False
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None


# ── Left-hand presets ─────────────────────────────────────────────────────────
class LeftHandPresets:
    def __init__(self, presets: dict, confirm_frames: int = 20,
                 cooldown: float = 4.0):
        self.presets        = presets
        self.confirm_frames = confirm_frames
        self.cooldown       = cooldown
        self._history: deque = deque(maxlen=confirm_frames)
        self._last_fired    = 0.0

    @staticmethod
    def _count_extended(lm) -> int:
        count = 0
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if lm[tip].y < lm[pip].y:
                count += 1
        return count

    def update(self, lm) -> Optional[str]:
        count = self._count_extended(lm)
        if count not in self.presets:
            self._history.clear()
            return None
        self._history.append(count)
        if len(self._history) < self.confirm_frames:
            return None
        if not all(c == count for c in self._history):
            return None
        now = time.time()
        if now - self._last_fired < self.cooldown:
            return None
        self._last_fired = now
        self._history.clear()
        return self.presets[count]

    def reset(self) -> None:
        self._history.clear()


# ── LSTM model ────────────────────────────────────────────────────────────────
class GestureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(63, HIDDEN, NUM_LAYERS, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASSES))
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ── Preprocessing ─────────────────────────────────────────────────────────────
def interpolate_frames(frames, target=TARGET_FRAMES):
    frames = np.array(frames, dtype=np.float32)
    T = len(frames)
    if T == target:
        return frames
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, 21, 3), dtype=np.float32)
    for j in range(21):
        for k in range(3):
            out[:, j, k] = np.interp(new_idx, old_idx, frames[:, j, k])
    return out


def normalize_landmarks(frames):
    frames = frames.copy()
    frames[:, :, :2] -= frames[:, 0:1, :2]
    scale = np.linalg.norm(
        frames[:, 9, :2] - frames[:, 0, :2], axis=-1, keepdims=True
    )
    scale = np.maximum(scale, 1e-6)
    frames[:, :, :2] /= scale[:, np.newaxis]
    return frames


def preprocess(buffer, device):
    frames = interpolate_frames(list(buffer))
    frames = normalize_landmarks(frames)
    x = frames.reshape(TARGET_FRAMES, -1)
    return torch.tensor(x).unsqueeze(0).to(device)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Presets ───────────────────────────────────────────────────────────────
    presets = parse_presets(args.presets) if args.presets else DEFAULT_PRESETS

    # ── TV connection ─────────────────────────────────────────────────────────
    tv = None
    if not args.no_tv:
        tv = WebOSRemote(
            host=args.webos_host,
            store_path=args.webos_store,
            secure=not args.webos_insecure,
        )
        try:
            tv.connect()
            print(f"[webOS] Connected to {args.webos_host}")

            if args.list_apps:
                apps = tv.list_apps()
                for title, app_id in sorted(apps.items()):
                    print(f"{app_id:50s}  {title}")
                tv.close()
                return

            tv.send("notify")
        except Exception as exc:
            print(f"[webOS] Could not connect: {exc}")
            print("Continuing in print-only mode.")
            tv = None
    else:
        if args.list_apps:
            print("Cannot list apps with --no-tv.")
            return

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GestureLSTM().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Model loaded on {device}")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    left_presets = LeftHandPresets(presets)

    # Right-hand inference state
    buffer       = deque(maxlen=TARGET_FRAMES)
    missed       = 0
    conf_history = {c: deque(maxlen=HISTORY_LEN) for c in CLASSES}
    last_trigger  = 0.0
    current_label = "..."
    current_conf  = 0.0

    print(f"\nRunning — press Q to quit")
    print(f"  webOS host : {args.webos_host if tv else 'not connected'}")
    print(f"  RIGHT hand : swipe/tap → TV arrows + OK")
    print(f"  LEFT hand  : hold fingers → apps {presets}\n")

    try:
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Split hands by screen position after flip:
                # left side of frame  = user's right hand (gestures)
                # right side of frame = user's left hand  (presets)
                right_lm = None
                left_lm  = None
                if results.multi_hand_landmarks:
                    for hand_obj, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        lm    = hand_obj.landmark
                        label = handedness.classification[0].label  # "Left" or "Right"
                        mp_draw.draw_landmarks(
                            frame, hand_obj, mp_hands.HAND_CONNECTIONS)
                        # After flip, MediaPipe's "Left" = user's right hand
                        if label == "Left":
                            left_lm = lm
                        else:
                            right_lm = lm

                # ── Right hand: buffer ────────────────────────────────────────
                if right_lm:
                    missed = 0
                    landmarks = [[lm.x, lm.y, lm.z] for lm in right_lm]
                    buffer.append(landmarks)
                else:
                    missed += 1
                    if missed > args.max_missed:
                        buffer.clear()
                        missed = 0
                        for hist in conf_history.values():
                            hist.clear()

                # ── Inference ─────────────────────────────────────────────────
                if len(buffer) == TARGET_FRAMES:
                    with torch.no_grad():
                        probs = torch.softmax(
                            model(preprocess(buffer, device)), dim=1
                        )[0].cpu().numpy()
                    for i, c in enumerate(CLASSES):
                        conf_history[c].append(float(probs[i]))

                # ── Peak detection ────────────────────────────────────────────
                now              = time.time()
                fired_this_frame = False

                if now - last_trigger > args.cooldown:
                    for c in CLASSES:
                        if c == "null" or fired_this_frame:
                            continue
                        signal = np.array(conf_history[c])
                        if len(signal) < args.peak_lag + 2:
                            continue
                        peaks, _ = find_peaks(
                            signal,
                            height=DEFAULT_CONFIDENCE[c],
                            prominence=args.prominence,
                            distance=PEAK_DISTANCE,
                        )
                        for p in peaks:
                            if p <= len(signal) - 1 - args.peak_lag:
                                current_label    = c
                                current_conf     = float(signal[p])
                                last_trigger     = now
                                fired_this_frame = True

                                cmd = GESTURE_TO_CMD.get(c)
                                print(f"-> {current_label}  ({current_conf:.0%})"
                                      f"  tv:{cmd}")
                                if cmd and tv:
                                    try:
                                        tv.send(cmd)
                                    except Exception as exc:
                                        print(f"[webOS] Send failed: {exc}")

                                for hist in conf_history.values():
                                    hist.clear()
                                buffer.clear()
                                break

                # ── Left hand: presets ────────────────────────────────────────
                if left_lm:
                    app_id = left_presets.update(left_lm)
                    if app_id:
                        print(f"-> preset: {app_id}")
                        if tv:
                            try:
                                tv.launch_app(app_id)
                            except Exception as exc:
                                print(f"[webOS] Launch failed: {exc}")
                else:
                    left_presets.reset()

                # ── Overlay ───────────────────────────────────────────────────
                elapsed         = now - last_trigger
                cooldown_active = elapsed < args.cooldown

                cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
                label_color = (0, 255, 128) if cooldown_active else (180, 180, 180)
                cv2.putText(frame, f"{current_label}  {current_conf:.0%}",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, label_color, 2)

                if cooldown_active:
                    bar_w = int(w * (1 - elapsed / args.cooldown))
                    cv2.rectangle(frame, (0, 55), (bar_w, 60),
                                  (0, 255, 128), -1)

                # Buffer fill
                buf_pct = len(buffer) / TARGET_FRAMES
                cv2.rectangle(frame, (0, h - 6),
                              (int(w * buf_pct), h), (80, 80, 80), -1)

                # Per-gesture confidence bars
                bar_h    = 40
                bar_y    = h - 6 - bar_h - 4
                bar_w_px = w // len(CLASSES)
                for i, c in enumerate(CLASSES):
                    if c == "null":
                        continue
                    val    = float(conf_history[c][-1]) \
                             if conf_history[c] else 0.0
                    bx     = i * bar_w_px
                    filled = int(bar_h * val)
                    color  = (0, 200, 100) \
                             if val >= DEFAULT_CONFIDENCE[c] else (60, 60, 60)
                    cv2.rectangle(frame,
                                  (bx, bar_y + bar_h - filled),
                                  (bx + bar_w_px - 2, bar_y + bar_h),
                                  color, -1)
                    cv2.putText(frame, c.replace("swipe_", "")[:3],
                                (bx + 4, bar_y + bar_h - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (160, 160, 160), 1)

                # Left-hand preset indicator
                if left_lm:
                    count = LeftHandPresets._count_extended(left_lm)
                    app   = presets.get(count, "")
                    cv2.putText(frame, f"L:{count}f  {app}",
                                (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (200, 180, 100), 2)

                cv2.imshow("Gesture TV Remote", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tv:
            tv.close()


if __name__ == "__main__":
    main()