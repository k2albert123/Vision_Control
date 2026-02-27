import asyncio
import websockets
import paho.mqtt.client as mqtt
import json

# Configuration
TEAM_NAME = "teamone"
MOVEMENT_TOPIC = f"vision/{TEAM_NAME}/movement"
HEARTBEAT_TOPIC = f"vision/{TEAM_NAME}/heartbeat"
BROKER_ADDRESS = "localhost"

clients_connected = set()

# ===== MQTT Callback Function =====
def handle_mqtt_message(client, userdata, message):
    payload_content = message.payload.decode()
    print(f"[MQTT] Received on {message.topic}: {payload_content}")
    # Schedule the broadcast coroutine in the event loop
    asyncio.run_coroutine_threadsafe(send_to_clients(payload_content), event_loop)

async def send_to_clients(msg_content):
    if not clients_connected:
        return
        
    active_sockets = clients_connected.copy()
    for sock in active_sockets:
        try:
            await sock.send(msg_content)
        except websockets.exceptions.ConnectionClosed:
            clients_connected.remove(sock)
        except Exception as e:
            print(f"[WS] Error sending to client: {e}")
            if sock in clients_connected:
                clients_connected.remove(sock)

async def websocket_handler(ws_connection):
    print(f"[WS] Client connected: {ws_connection.remote_address}")
    clients_connected.add(ws_connection)
    try:
        await ws_connection.wait_closed()
    except Exception as e:
        print(f"[WS] Connection error: {e}")
    finally:
        print(f"[WS] Client disconnected: {ws_connection.remote_address}")
        clients_connected.remove(ws_connection)

# ===== MQTT Setup =====
mqtt_listener = mqtt.Client()
mqtt_listener.on_message = handle_mqtt_message
try:
    mqtt_listener.connect(BROKER_ADDRESS, 1883)
    mqtt_listener.subscribe([(MOVEMENT_TOPIC, 0), (HEARTBEAT_TOPIC, 0)])
    print(f"[MQTT] Connected to {BROKER_ADDRESS}:1883")
    print(f"[MQTT] Subscribed to {MOVEMENT_TOPIC} and {HEARTBEAT_TOPIC}")
    mqtt_listener.loop_start()
except Exception as e:
    print(f"[MQTT] Connection failed: {e}")

# ===== Asyncio Server Loop =====
async def start_server():
    print("WS Server starting on ws://0.0.0.0:9002")
    async with websockets.serve(websocket_handler, "0.0.0.0", 9002):
        print("WS Server active at ws://0.0.0.0:9002")
        await asyncio.Future()  # Keep running indefinitely

# ===== Execution Entry Point =====
if __name__ == "__main__":
    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(start_server())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        mqtt_listener.loop_stop()
        mqtt_listener.disconnect()
