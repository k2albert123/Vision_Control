import network
import time
from umqtt.simple import MQTTClient
from machine import Pin, PWM
import json

# Network Credentials
WIFI_SSID = "Wireless1"
WIFI_PASS = "@RcaNyabihu2023"

# Message Broker Config
TEAM_NAME = "teamone"
SERVER_IP = "10.12.73.101"
SUB_TOPIC = b"vision/teamone/movement"
PUB_HEARTBEAT = b"vision/teamone/heartbeat"

# Servo Motor Initialization
servo_motor = PWM(Pin(5), freq=50)
current_position = 90

# Tuning parameters for Phase 2 proportional control
Kp = 0.05
MAX_STEP = 5

def update_servo_position(deg):
    # Calculate duty cycle for the given degree (40 to 115 roughly maps to 0 to 180)
    duty_cycle = int(40 + (deg / 180) * 75)
    servo_motor.duty(duty_cycle)

update_servo_position(current_position)

def on_mqtt_message(topic, message):
    global current_position
    try:
        payload = json.loads(message)
    except ValueError:
        print("Invalid JSON received")
        return

    command = payload.get("status")
    error = payload.get("error", 0)

    if command == "MOVE_LEFT":
        # Check if we have error for proportional control
        if error != 0:
            step = min(MAX_STEP, abs(error) * Kp)
            current_position -= step
        else:
            current_position -= 2
    elif command == "MOVE_RIGHT":
        if error != 0:
            step = min(MAX_STEP, abs(error) * Kp)
            current_position += step
        else:
            current_position += 2

    # Clamp the angle between 0 and 180
    current_position = max(0, min(180, current_position))
    update_servo_position(current_position)
    print(f"Setting servo to {current_position}")

# Connection logic
wlan_interface = network.WLAN(network.STA_IF)
wlan_interface.active(True)

def connect_wifi():
    if not wlan_interface.isconnected():
        print(f"Connecting to {WIFI_SSID}...")
        wlan_interface.connect(WIFI_SSID, WIFI_PASS)
        while not wlan_interface.isconnected():
            time.sleep(0.5)
            print(".", end="")
        print(f"\nConnected: {wlan_interface.ifconfig()}")

def connect_mqtt():
    global mqtt_client
    while True:
        try:
            print(f"Connecting to MQTT broker at {SERVER_IP}...")
            mqtt_client = MQTTClient("esp8266_client", SERVER_IP)
            mqtt_client.set_callback(on_mqtt_message)
            mqtt_client.connect()
            mqtt_client.subscribe(SUB_TOPIC)
            print(f"Subscribed to {SUB_TOPIC}")
            return
        except Exception as e:
            print(f"Failed to connect to MQTT: {e}. Retrying in 5s...")
            time.sleep(5)

connect_wifi()
connect_mqtt()

last_heartbeat = time.time()

while True:
    try:
        # Reconnect if WiFi dropped
        if not wlan_interface.isconnected():
            connect_wifi()
            connect_mqtt()

        mqtt_client.check_msg()

        # Publish heartbeat every 5 seconds
        if time.time() - last_heartbeat > 5:
            heartbeat_payload = json.dumps({
                "node": "esp8266",
                "status": "ONLINE",
                "timestamp": time.time()
            })
            mqtt_client.publish(PUB_HEARTBEAT, heartbeat_payload)
            last_heartbeat = time.time()

    except Exception as e:
        print(f"MQTT Error: {e}. Reconnecting...")
        time.sleep(2)
        connect_wifi()
        connect_mqtt()
    
    time.sleep(0.1)

