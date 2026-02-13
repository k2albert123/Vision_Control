Distributed Vision-Control System (Face-Locked Servo)
📌 1. System Description

The Distributed Vision-Control System is a real-time face-tracking platform built using a distributed architecture.

The system detects a human face through a PC camera and adjusts a servo motor to keep the face centered in the frame. Communication between components is handled using MQTT and WebSocket protocols.

🎯 How It Works

The Vision Node (PC) captures video frames.

The system detects a face using OpenCV.

Based on face position, it determines movement:

MOVE_LEFT

MOVE_RIGHT

CENTERED

NO_FACE

The movement command is published via MQTT.

The ESP8266 receives the MQTT message and rotates the servo.

The Backend relays MQTT updates to the web dashboard using WebSocket.

The Dashboard displays live tracking data.

🏗 System Architecture
[ PC - Vision Node ]
        |
        | MQTT (vision/<team_id>/movement)
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

🔑 Core Communication Rule

Vision Node → Publishes via MQTT

ESP8266 → Subscribes via MQTT

Backend → Subscribes via MQTT

Dashboard → Connects via WebSocket

MQTT Broker → Routes messages

There are no direct connections between:

PC ↔ ESP8266

Dashboard ↔ MQTT

📡 2. MQTT Topics Used

Each team must define a unique team ID:

TEAM_NAME = "TeAmOnE"

Primary Movement Topic
vision/TeAmOnE/movement

Message Format Example
{
  "status": "MOVE_LEFT",
  "confidence": 0.9,
  "timestamp": 1730000000
}

Optional Heartbeat Topic
vision/TeAmOnE/heartbeat


Example:

{
  "node": "pc",
  "status": "ONLINE",
  "timestamp": 1730000000
}


⚠️ Important:

Do NOT use wildcard topics.

Do NOT subscribe to other teams’ topics.

Each team must isolate its topic namespace.

🌐 3. Live Dashboard URL

The WebSocket server runs locally on:

ws://localhost:9002


The live dashboard is accessed by opening:

dashboard/index.html


Make sure the WebSocket connection inside index.html is:

const socket = new WebSocket("ws://localhost:9002");


When running locally, the dashboard will display:

Current movement status

Detection confidence

Timestamp of last update

📁 Project Structure
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
└── README.md

⚙️ Setup Instructions
1️⃣ Install Dependencies (PC)
pip install opencv-python paho-mqtt websockets asyncio


Requirements:

Python 3.10+

OpenCV

Paho-MQTT

WebSockets

2️⃣ Install MQTT Broker (Mosquitto)
Windows

Run:

mosquitto.exe -v

Linux
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto

3️⃣ Running the System
Step 1 – Start MQTT Broker

Windows:

mosquitto -v


Linux:

sudo systemctl start mosquitto

Step 2 – Start Backend
cd backend
python backend.py


You should see:

WS Server active at ws://0.0.0.0:9002

Step 3 – Run Vision Node
cd vision-node
python vision_node.py


The camera window will open and start publishing MQTT messages.

Step 4 – Open Dashboard

Open:

dashboard/index.html


The dashboard connects to:

ws://localhost:9002

Step 5 – Configure ESP8266

Install MicroPython.

Update broker IP to your PC’s local IP.

Connect servo to GPIO5 (D1).

Upload and run main.py.

🚀 Features

Real-time face tracking

Distributed MQTT-based architecture

Live WebSocket dashboard

Topic isolation for multi-team environments

Local-only deployment (no VPS required)

Supports open-loop and closed-loop tracking

🏁 Operational Flow Summary

Camera detects face
→ Movement command computed
→ MQTT message published
→ Broker forwards message
→ ESP8266 rotates servo
→ Backend relays update
→ Dashboard updates in real time