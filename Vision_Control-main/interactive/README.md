# Distributed Vision-Control System (Face-Locked Servo)

📌 **1. System Description**

The Distributed Vision-Control System is a real-time face-tracking platform built using a distributed architecture.
The system detects a human face through a PC camera and adjusts a servo motor to keep the face centered in the frame. Communication between components is handled using MQTT and WebSocket protocols, alongside direct physical USB connections for advanced face locking.

### 🎯 How It Works

- The **Vision Node (PC)** captures video frames and tracks faces.
- **[Phase 3]** The system recognizes an enrolled face using an `LBPH` model.
- Based on the enrolled face position, it determines movement: `MOVE_LEFT`, `MOVE_RIGHT`, `CENTERED`, `NO_FACE`.
- It calculates an `error` offset for proportional control.
- **[Phase 3]** The error is sent directly via **USB Serial** to an Arduino running a tracking sketch.
- The movement command and error are additionally published via MQTT.
- The **ESP8266** receives the MQTT message and also proportionally rotates the servo (if used in a fully wireless setup).
- The **Backend** relays MQTT updates to the web dashboard using WebSocket.
- The **Dashboard** displays live tracking data, confidence, and system heartbeats.

### 🏗 System Architecture

```text
[ PC - Vision Node (LBPH Face Recognition) ]
        |                   |
        | USB Serial        | MQTT (vision/teamone/movement & vision/teamone/heartbeat)
        v                   v
[ Arduino Controller ]  [ MQTT Broker ]
        |                   |
        | PWM               | MQTT
        v                   v
[ Tracker Servo ]       [ Backend WebSocket Relay ]
                            |
                            | WebSocket (ws://localhost:9002)
                            v
                        [ Web Dashboard ]
```

### 🔑 Core Communication Rule

- Vision Node → Publishes via MQTT & transmits USB Serial for local motor tracking.
- Arduino → Parses Serial integer inputs to drive motor physical tracking.
- ESP8266/Backend → Subscribes via MQTT.
- Dashboard → Connects via WebSocket.
- MQTT Broker → Routes messages.

📡 **2. MQTT Topics Used**

Each team must strictly isolate its topic namespace.

`TEAM_NAME = "teamone"`

**Primary Movement Topic**

```text
vision/teamone/movement
```

Message Format Example

```json
{
  "status": "MOVE_LEFT",
  "confidence": 0.9,
  "error": -65,
  "timestamp": 1730000000
}
```

**Heartbeat Topic**

```text
vision/teamone/heartbeat
```

Example:

```json
{
  "node": "pc",
  "status": "ONLINE",
  "timestamp": 1730000000
}
```

⚠️ **Important:** Do NOT use wildcard topics. Do NOT subscribe to other teams’ topics.

🌐 **3. Live Dashboard URL**

The WebSocket server runs locally on: `ws://localhost:9002`

The live dashboard is accessed by opening: `dashboard/index.html`

📁 **Project Structure**

```text
distributed-vision-control/
│
├── vision-node/
│   ├── vision_node.py
│   └── enroll.py          <-- [New] Run to capture tracking profile
│
├── backend/
│   └── backend.py
│
├── esp8266/
│   └── main.py
│
├── arduino/
│   └── servo_control.ino  <-- [New] Upload to physical USB-connected Arduino
│
├── dashboard/
│   └── index.html
│
├── mosquitto/
│   └── mosquitto.conf
│
├── requirements.txt
└── README.md
```

⚙️ **Setup Instructions**

**1️⃣ Install Dependencies (PC)**
Make sure to replace basic OpenCV with the `contrib` library for the LBPH recognizer.

```bash
pip uninstall opencv-python
pip install -r requirements.txt
```

**2️⃣ Hardware Setup (Arduino Tracking)**

1. Connect an Arduino using a USB cable.
2. Ensure the Serial Port in `vision_node.py` matches the Arduino (e.g. `COM3` or `/dev/ttyUSB0`).
3. Connect the Servo signal pin to Arduino **D9**.
4. Upload `arduino/servo_control.ino` using the Arduino IDE.

**3️⃣ Running the System**

**Step 1 – Enroll Your Face**
We use LBPH offline tracking to lock your face identity. Look at the camera.

```bash
cd vision-node
python enroll.py
```

**Step 2 – Start MQTT Broker**

```bash
mosquitto -c mosquitto/mosquitto.conf -v
```

**Step 3 – Start Backend**

```bash
cd backend
python backend.py
```

**Step 4 – Run Vision Node**
Wait for the camera window. When it locks your face, the Arduino servo will physically track you horizontally, and UI updates will broadcast.

```bash
cd vision-node
python vision_node.py
```

**Step 5 – Open Dashboard**
Open `dashboard/index.html` in any modern web browser to view the real-time MQTT feed.

🚀 **Features**

- **Offline Face-Locking**: Uses efficient `LBPH` histograms to recognize and lock only the target user.
- **Low Latency Tracking Engine**: Relays target offsets via USB `pyserial` directly to Arduino.
- Distributed MQTT-based architecture with strict topic isolation.
- Live WebSocket dashboard showing statuses and system heartbeat logs.
