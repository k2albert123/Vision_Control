import asyncio
import websockets
import paho.mqtt.client as mqtt
import json

# Configuration
TEAM_NAME = "TeAmOnE"
TOPIC_NAME = f"vision/{TEAM_NAME}/movement"
BROKER_ADDRESS = "localhost"

clients_connected = set()

# ===== MQTT Callback Function =====
def handle_mqtt_message(client, userdata, message):
    payload_content = message.payload.decode()
    # Schedule the broadcast coroutine in the event loop
    asyncio.run_coroutine_threadsafe(send_to_clients(payload_content), event_loop)

async def send_to_clients(msg_content):
    active_sockets = clients_connected.copy()
    for sock in active_sockets:
        try:
            await sock.send(msg_content)
        except:
            clients_connected.remove(sock)

async def websocket_handler(ws_connection):
    clients_connected.add(ws_connection)
    try:
        await ws_connection.wait_closed()
    finally:
        clients_connected.remove(ws_connection)

# ===== MQTT Setup =====
mqtt_listener = mqtt.Client()
mqtt_listener.on_message = handle_mqtt_message
mqtt_listener.connect(BROKER_ADDRESS, 1883)
mqtt_listener.subscribe(TOPIC_NAME)
mqtt_listener.loop_start()

# ===== Asyncio Server Loop =====
async def start_server():
    async with websockets.serve(websocket_handler, "0.0.0.0", 9002):
        print("WS Server active at ws://0.0.0.0:9002")
        await asyncio.Future()  # Keep running indefinitely

# ===== Execution Entry Point =====
event_loop = asyncio.get_event_loop()
event_loop.run_until_complete(start_server())
