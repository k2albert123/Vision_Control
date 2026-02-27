# Face Recognition & Face Locking System (Windows)

A **real-time face recognition and face locking system** that runs on **Windows + Python 3.12**.

Pipeline: **Camera → Haar detection → FaceMesh 5-point landmarks → Face alignment (112×112) → ArcFace embedding**

Key capabilities:
- Detect and recognize multiple faces simultaneously
- **Lock onto a specific target face** with identity-based tracking + spatial fallback
- Detect actions on the locked face (head movement, smile)
- Log all events to a `.jsonl` history file
- Control a **servo motor** via **MQTT over ESP8266** — the servo tracks the locked face's horizontal position

---

## 📁 Project Structure

```
Facelocking2/
│
├── .venv/
├── data/
│   ├── db/
│   │   ├── face_db.npz          # Enrolled face embeddings
│   │   └── face_db.json         # DB metadata
│   ├── enroll/                  # Saved crop images per person
│   └── history/
│       └── history_log.jsonl    # Action event log (JSONL)
│
├── models/
│   └── embedder_arcface.onnx    # ArcFace ONNX model
│
├── src/
│   ├── __init__.py
│   ├── detect.py                # Main detection + face locking loop
│   ├── enroll.py                # Face enrollment tool
│   ├── faceLockServo.py         # Face locking + MQTT servo control
│   ├── face_locking_controller.py
│   ├── face_tracker.py
│   ├── haar_5pt.py              # Haar + MediaPipe FaceMesh detector
│   ├── align.py                 # 5-point face alignment
│   ├── embed.py                 # ArcFace ONNX embedder
│   ├── recognize.py             # DB matching / recognition helpers
│   ├── action_detector.py       # Head movement & smile detection
│   ├── history_manager.py       # JSONL event logger
│   ├── history_logger.py
│   ├── actions.py
│   ├── landmarks.py
│   ├── camera.py
│   ├── evaluate.py
│   └── servo_controller/
│       └── servo_controller.ino # ESP8266 Arduino sketch
│
├── init_project.py
└── README.md
```

---

## 🐍 Python Version

```
Python 3.12.4
```

---

## 🔧 Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install opencv-python numpy onnxruntime mediapipe insightface paho-mqtt
```

### MediaPipe version fix

```powershell
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

---

## 🧠 ArcFace Model

Copy the model from the InsightFace buffalo pack:

```powershell
Copy-Item buffalo_l\w600k_r50.onnx models\embedder_arcface.onnx
```

---

## 👤 Step 1 — Enroll a Face

```powershell
python -m src.enroll
```

**Controls during enrollment:**

| Key | Action |
|-----|--------|
| `SPACE` | Capture one sample |
| `a` | Toggle auto-capture (every 0.25 s) |
| `s` | Save all samples to DB |
| `r` | Reset new samples (keep existing) |
| `q` | Quit |

- Needs **15 samples** by default (`EnrollConfig.samples_needed`)
- Saves aligned 112×112 crops to `data/enroll/<name>/`
- Stores the mean ArcFace embedding in `data/db/face_db.npz`

---

## ▶️ Step 2 — Run Face Locking (detect.py)

```powershell
python -m src.detect
```

What this does:
1. Opens the camera (tries indices 0 → 1 → 2)
2. Detects all faces each frame using `HaarFaceMesh5pt`
3. Embeds and matches every face against the DB (`dist_thresh=0.34`)
4. **Locks onto "Kheira"** (hardcoded target — change `r["name"] == "Kheira"` in `detect.py` to your enrolled name)
5. Tracks the locked face with identity matching + spatial fallback (≤ 50 px keypoint distance)
6. Releases lock after **2 seconds** of not seeing the target
7. Runs `ActionDetector` on the locked face — detects head movement (left/right/up/down) and smile
8. Logs all events via `HistoryManager` → `data/history/history_log.jsonl`

**Color coding:**

| Color | Meaning |
|-------|---------|
| 🟡 Yellow | Locked target face |
| 🟢 Green | Recognised other face |
| 🔴 Red | Unknown face |

Press `q` to quit.

---

## 🎯 Step 3 (Optional) — Run with Servo Control (faceLockServo.py)

```powershell
python src/faceLockServo.py
```

- Prompts you to choose a target identity from the enrolled DB
- Locks face and publishes **servo angles (0–180°)** to MQTT topic `TeAmSiX/facelocking/servo_ctrl_x9z`
- MQTT Broker: `157.173.101.159:1883`
- Uses **MediaPipe FaceMesh** (full 468 landmarks) for detection in this mode
- Servo angle is smoothed and rate-limited (deadzone: 5°, interval: 100 ms)

---

## 🤖 ESP8266 Servo Controller

File: `src/servo_controller/servo_controller.ino`

- Connects to WiFi: `Main Hall`
- Subscribes to MQTT topic: `TeAmSiX/facelocking/servo_ctrl_x9z`
- Servo on pin `D1`, range 0–180°
- **Search mode**: if no MQTT message arrives for **1500 ms**, the servo sweeps back and forth automatically
- Smooth movement: increments target angle by 1° per `loop()` tick (15 ms delay)

**Dependencies (Arduino Library Manager):**
- `ESP8266WiFi`
- `PubSubClient`
- `Servo`

---

## 📝 Action Detection

`ActionDetector` runs on the locked face every frame using the 5-point keypoints `[left_eye, right_eye, nose, left_mouth, right_mouth]`:

| Action | Detection method |
|--------|-----------------|
| `moved left/right/up/down` | Bounding box centre displacement > 15 px |
| `smile` | Mouth width / face width ratio crosses 0.40 threshold |

Events are written immediately to `data/history/history_log.jsonl` (JSONL format).

---

## ❗ Common Errors

**`ModuleNotFoundError`**
- Make sure `src/__init__.py` exists
- Always run with `python -m src.detect` (not `python src/detect.py`)

**MediaPipe import error**
```powershell
pip install mediapipe==0.10.9
```

**`Failed to open camera`**
- `enroll.py` and `detect.py` try camera indices 0, 1, 2 automatically
- Make sure no other application is using the camera

**MQTT connection timeout**
- Check that the broker at `157.173.101.159:1883` is reachable from your network
- `faceLockServo.py` will still run in offline mode (servo commands will not be sent)

---

## 🔒 Changing the Lock Target

In `src/detect.py`, line 92:
```python
candidates = [r for r in recognized_faces if r["accepted"] and r["name"] == "Albert"]
```
Replace `"Albert"` with any name you have enrolled.

---

## 🚀 Possible Extensions

- FAISS approximate nearest-neighbour search for large DBs
- Full blink detection (requires 68-point or FaceMesh EAR landmarks)
- GUI dashboard
- Multi-target locking
- Cloud/database event logging
