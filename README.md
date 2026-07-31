# Hands Remote Control

A hand-gesture remote control for an LG webOS TV. Right hand swipes/taps drive navigation (up/down/left/right/OK); left hand finger-counts launch apps (hold up 2 fingers to open YouTube, 1 for Netflix, etc.). No physical remote, no wearables — just a webcam, MediaPipe hand tracking, and a small LSTM trained on self-collected gesture data.

> **Status:** working, and the most complete project reviewed in terms of end-to-end pipeline — this isn't just inference code, it's the full loop: a data-collection tool, a training notebook, a trained model, real-time inference, and a GUI on top. One small cleanup item below.

---

## How it works

1. **`data_collection.py`** — a hands-on labeling tool: hold a key (`r`/`l`/`u`/`d`/`t`) to tag the next 15 frames of hand-landmark data as that gesture, with a live progress bar and running sample counts. This is how the training set (roughly 33–37 samples per swipe/tap class, 160 for "no gesture") was actually built.
2. **`trainning.ipynb`** — trains a 2-layer LSTM on the collected landmark sequences: interpolates every sequence to a fixed 15 frames (rather than truncating or padding, which would cut off or dilute the gesture), normalizes each frame to be wrist-relative and scale-invariant (so it works regardless of hand size or distance from camera), balances classes with a weighted sampler, and trains with early stopping and LR scheduling.
3. **`main.py`** — the real-time application: tracks both hands simultaneously via MediaPipe, feeds a rolling 15-frame buffer of the right hand's landmarks through the trained LSTM every frame, and uses **peak detection** (via `scipy.signal.find_peaks`) on the resulting confidence stream to fire a gesture exactly once at its most confident moment — rather than firing repeatedly while the gesture is happening or missing it because a single frame threshold wasn't crossed. The left hand runs a much simpler finger-counting classifier that triggers app launches after a hold-and-confirm period, so it can't be triggered by a hand passing through frame.
4. Gestures/presets are translated into LG webOS commands over a WebSocket (`pywebostv`), with automatic pairing, input-socket reconnection on failure, and full app listing/launching support.
5. **`gui.py`** — a Tkinter control panel that scans the local subnet for a webOS TV (port 3000), fetches its installed app list, and lets you configure left-hand presets visually instead of hand-editing a command-line string.

---

## Skills demonstrated

**End-to-end ML pipeline, not just a downloaded model**
- Self-built dataset from scratch (custom data collection tool → labeled JSON samples → training notebook → deployed model) — every stage of the ML lifecycle is represented, not just "call an API"
- Class imbalance handled properly with a `WeightedRandomSampler` rather than ignored
- Train/inference consistency enforced deliberately: the exact same `interpolate_frames`/`normalize_landmarks` functions are duplicated verbatim between the notebook and `main.py`, with a comment calling out that they *must* match — a real, easy-to-get-wrong detail in ML deployment that's been explicitly guarded against

**Real-time signal processing**
- Using peak detection on a live confidence stream (rather than a naive "did confidence cross a threshold this frame" check) to correctly fire one event per physical gesture — this is the kind of detail that separates a demo from something usable day to day
- Cooldown windows and missed-frame buffer resets to avoid double-firing or getting stuck on stale hand data

**Computer vision**
- Dual simultaneous hand tracking with correct left/right disambiguation after a camera-mirror flip (a detail that's easy to get backwards)
- Two different gesture-recognition strategies used deliberately for two different needs: an LSTM for dynamic swipe motions (which unfold over time), and simple geometric finger-counting for static hand poses (which don't need a trained model at all) — right tool for each sub-problem instead of one model for everything

**Systems integration**
- Full LG webOS protocol integration (`pywebostv`): pairing/registration flow, input socket reconnect-on-failure, media/system/application control layers, and graceful degradation to "print-only mode" if the TV can't be reached
- A GUI layer with concurrent network scanning (`ThreadPoolExecutor` probing an entire subnet in parallel) to auto-discover the TV rather than requiring a hardcoded IP

---

## Tech stack

| Layer | Tools |
|---|---|
| Hand tracking | MediaPipe |
| Model | PyTorch (LSTM) |
| Signal processing | SciPy (`find_peaks`) |
| TV control | pywebostv (LG webOS WebSocket API) |
| Vision I/O | OpenCV |
| GUI | Tkinter |

---

## Getting started

```bash
pip install opencv-python mediapipe numpy torch scipy pywebostv
```

Collect your own gesture data (optional — a trained `gesture_model.pt` is already included):
```bash
python data_collection.py
# hold r/l/u/d/t to label swipe_right/left/up/down/tap, anything else = "null"
```

Retrain if you collect new data: open `trainning.ipynb` and run all cells — it saves `gesture_model.pt`.

Run the remote:
```bash
python main.py --webos-host 10.0.0.6
# first run: accept the pairing prompt on the TV
```

Or launch the GUI to auto-discover the TV and configure app presets:
```bash
python gui.py
```

---

## Roadmap / known limitations

- **Small training set**: ~33–37 samples per gesture class is enough to get a working demo but is thin for robustness across different lighting, hand sizes, or camera angles — more data collection would meaningfully improve reliability.
- **Single-user calibration**: normalization handles scale and position, but the model was trained on one person's gesture style; a second user's swipe timing/shape might need either more diverse training data or a short recalibration step.
- **No automated tests**: correctness has been validated by watching the live confidence bars and TV response, which is a reasonable way to tune a real-time gesture system, but not something that catches regressions automatically.

Everything else here already works end-to-end: this is a complete, personally-collected-and-trained gesture recognition system controlling a real device over its native protocol — a strong project for demonstrating applied ML and systems integration together, not just one or the other.
