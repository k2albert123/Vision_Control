import cv2
import paho.mqtt.client as mqtt
import json
import time
import threading

# ========== CONFIG PARAMETERS ==========
TEAM_NAME = "teamone"
MQTT_SERVER = "10.12.73.101"
MQTT_PORT = 1883

MOVEMENT_TOPIC = f"vision/{TEAM_NAME}/movement"
HEARTBEAT_TOPIC = f"vision/{TEAM_NAME}/heartbeat"

# Setup MQTT Client
mqtt_publisher = mqtt.Client()
mqtt_publisher.connect(MQTT_SERVER, MQTT_PORT, 60)
mqtt_publisher.loop_start()

# Initialize Head Detection Model
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

video_stream = cv2.VideoCapture(0)

# State Tracking
last_published_status = None
last_published_error = None
ERROR_THRESHOLD = 15 # Publish again if error changes by this much (for proportional control)

def determine_movement_command(face_x_pos, width_of_frame):
    midpoint = width_of_frame // 2
    dead_zone = 50
    error = face_x_pos - midpoint

    if error < -dead_zone:
        return "MOVE_LEFT", error
    elif error > dead_zone:
        return "MOVE_RIGHT", error
    else:
        return "CENTERED", error

def publish_heartbeat():
    while True:
        payload = {
            "node": "pc",
            "status": "ONLINE",
            "timestamp": int(time.time())
        }
        mqtt_publisher.publish(HEARTBEAT_TOPIC, json.dumps(payload))
        time.sleep(5)  # Every 5 seconds

# Start heartbeat thread
threading.Thread(target=publish_heartbeat, daemon=True).start()

print(f"[VISION] Started. Publishing to {MOVEMENT_TOPIC}")

try:
    while True:
        success, current_frame = video_stream.read()
        if not success:
            break
            
        grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(grayscale_frame, 1.3, 5)

        current_status = "NO_FACE"
        detection_certainty = 0.0
        error = 0

        if len(detected_faces) > 0:
            (x_pos, y_pos, w_size, h_size) = detected_faces[0]
            center_x = x_pos + w_size // 2
            current_status, error = determine_movement_command(center_x, current_frame.shape[1])
            detection_certainty = 0.9

            # Draw bounding box
            cv2.rectangle(current_frame, (x_pos, y_pos), (x_pos+w_size, y_pos+h_size), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(current_frame, (center_x, y_pos + h_size//2), 4, (0, 0, 255), -1)

        # Logic to "avoid flooding" - only publish when State changes or Error changes significantly
        should_publish = False
        
        if current_status != last_published_status:
            should_publish = True
        elif current_status in ["MOVE_LEFT", "MOVE_RIGHT"] and last_published_error is not None:
             if abs(error - last_published_error) > ERROR_THRESHOLD:
                 should_publish = True

        if should_publish:
            msg_payload = {
                "status": current_status,
                "confidence": detection_certainty,
                "error": error,
                "timestamp": int(time.time())
            }
            mqtt_publisher.publish(MOVEMENT_TOPIC, json.dumps(msg_payload))
            print(f"[VISION] Published: {current_status} | Error: {error}")
            
            last_published_status = current_status
            last_published_error = error

        cv2.imshow("Surveillance Feed", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_stream.release()
    cv2.destroyAllWindows()
    mqtt_publisher.loop_stop()
    mqtt_publisher.disconnect()

