import network
import time
from umqtt.simple import MQTTClient
from machine import Pin, PWM
import json

# Network Credentials
WIFI_SSID = "Wireless1"
WIFI_PASS = "@RcaNyabihu2023"

# Message Broker Config
GROUP_ID = "TeAmOnE"
SERVER_IP = "10.12.73.101"
SUB_TOPIC = b"vision/TeAmOnE/movement"

# Initialize WiFi Connection
wlan_interface = network.WLAN(network.STA_IF)
wlan_interface.active(True)
wlan_interface.connect(WIFI_SSID, WIFI_PASS)

while not wlan_interface.isconnected():
    pass

# Servo Motor Initialization
servo_motor = PWM(Pin(5), freq=50)

current_position = 90

def update_servo_position(deg):
    # Calculate duty cycle for the given degree
    duty_cycle = int(40 + (deg / 180) * 75)
    servo_motor.duty(duty_cycle)

update_servo_position(current_position)

def on_mqtt_message(topic, message):
    global current_position
    try:
        payload = json.loads(message)
    except ValueError:
        return

    command = payload.get("status")

    if command == "MOVE_LEFT":
        current_position -= 5
    elif command == "MOVE_RIGHT":
        current_position += 5

    # Clamp the angle between 0 and 180
    current_position = max(0, min(180, current_position))
    update_servo_position(current_position)

mqtt_subscriber = MQTTClient("esp8266_client", SERVER_IP)
mqtt_subscriber.set_callback(on_mqtt_message)
mqtt_subscriber.connect()
mqtt_subscriber.subscribe(SUB_TOPIC)

while True:
    mqtt_subscriber.check_msg()
    time.sleep(0.1)
