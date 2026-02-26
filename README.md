# Distributed Vision-Control System (Face-Locked Servo)

📌 **1. System Description**

The Distributed Vision-Control System is a real-time face-tracking platform built using a distributed architecture.
The system detects a human face through a PC camera and adjusts a servo motor to keep the face centered in the frame. Communication between components is handled using MQTT and WebSocket protocols.

### 🎯 How It Works

*   The **Vision Node (PC)** captures video frames and detects a face using OpenCV.
*   Based on face position, it determines movement: `MOVE_LEFT`, `MOVE_RIGHT`, `CENTERED`, `NO_FACE`.
*   It calculates an `error` offset for proportional control (Phase 2).
*   The movement command and error are published via MQTT.
*   The **ESP8266** receives the MQTT message and proportionally rotates the servo.
*   The **Backend** relays MQTT updates to the web dashboard using WebSocket.
*   The **Dashboard** displays live tracking data, confidence, and system heartbeats.

### 🏗 System Architecture

```text
[ PC - Vision Node ]
        |
        | MQTT (vision/teamone/movement & vision/teamone/heartbeat)
        v
[ MQTT Broker ]
        |
        | MQTT
        v
[ ESP8266 Edge Controller ] ---> [ Servo Motor ]
        |
        | MQTT
        v
[ Backend WebSocket Relay ]
        |
        | WebSocket (ws://localhost:9002)
        v
[ Web Dashboard ]
```

### 🔑 Core Communication Rule

*   Vision Node → Publishes via MQTT
*   ESP8266 → Subscribes via MQTT
*   Backend → Subscribes via MQTT
*   Dashboard → Connects via WebSocket
*   MQTT Broker → Routes messages
*   There are **no direct connections** between PC ↔ ESP8266 or Dashboard ↔ MQTT.

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
│   └── vision_node.py
│
├── backend/
│   └── backend.py
│
├── esp8266/
│   └── main.py
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
```bash
pip install -r requirements.txt
```

**2️⃣ Install MQTT Broker (Mosquitto)**

**Windows**
Run Mosquitto using the provided config:
```bash
mosquitto -c mosquitto/mosquitto.conf -v
```

**Linux**
```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

**3️⃣ Running the System**

**Step 1 – Start MQTT Broker**
(See above)

**Step 2 – Start Backend**
```bash
cd backend
python backend.py
```
You should see: `WS Server active at ws://0.0.0.0:9002`

**Step 3 – Run Vision Node**
```bash
cd vision-node
python vision_node.py
```
The camera window will open and start publishing MQTT messages.

**Step 4 – Open Dashboard**
Open `dashboard/index.html` in any modern web browser.

**Step 5 – Configure ESP8266**
*   Install MicroPython.
*   Update broker IP and WiFi credentials in `main.py`.
*   Connect servo to GPIO5 (D1).
*   Upload and run `main.py`.

🚀 **Features**

*   Real-time face tracking with Proportional Control (Phase 2 capability).
*   Distributed MQTT-based architecture with strict topic isolation.
*   Live WebSocket dashboard showing statuses and system heartbeat logs.
*   Robust connection logic for headless Edge modules (ESP8266).