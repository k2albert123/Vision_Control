# Vision Control Projects

This repository contains two distinct yet related projects focused on real-time face recognition and target-locking with servo motor control.

## 1. Main System (ArcFace + MediaPipe + MQTT)

**Location:** `./Vision_Control-main/README.md`

A robust Python 3.12 application that uses a multi-stage pipeline (Haar detection → MediaPipe FaceMesh landmarks → ArcFace embeddings) to detect, recognize, and lock onto a specific target's face.

**Key Features:**

- Enroll users into a database using 112×112 cropped images.
- Recognize faces in real-time and lock onto a configured target.
- Action detection: Detects head movements, smiles, and logs events.
- Servo Control: Publishes target tracking angles via MQTT to an ESP8266 controller, allowing a physical servo to track the locked face.

## 2. Interactive / Distributed System (LBPH + WebSockets)

**Location:** `./Vision_Control/interactive/README.md`

A distributed architecture designed for low-latency target tracking and live dashboard visualization. It uses an offline LBPH recognizer to track an enrolled face.

**Key Features:**

- Direct USB Serial communication to an Arduino for fast, proportional physical servo tracking.
- Parallel MQTT-based architecture for broadcasting tracking errors and system heartbeats.
- Real-time WebSocket relay to a live web dashboard showing active tracking data, confidence metrics, and node statuses.
- Strict topic isolation making it suitable for multi-team or competition environments (`vision/teamone/...`).

---

### Setup and Execution

To get started, navigate to the specific project directory and follow the instructions in the corresponding `README.md`.

- For ArcFace-based Face Locking:

  ```bash
  cd Vision_Control
  ```

- For LBPH Distributed Control & Dashboard:
  ```bash
  cd Vision_Control/interactive
  ```
