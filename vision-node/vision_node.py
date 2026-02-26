import cv2
import paho.mqtt.client as mqtt
import json
import time
import threading
import serial
import os

# ========== CONFIG PARAMETERS ==========
TEAM_NAME = "teamone"
MQTT_SERVER = "10.12.73.101"
MQTT_PORT = 1883

MOVEMENT_TOPIC = f"vision/{TEAM_NAME}/movement"
HEARTBEAT_TOPIC = f"vision/{TEAM_NAME}/heartbeat"

# USB Serial Configuration (Update to match your Arduino Port)
SERIAL_PORT = "COM3" # Windows: COM3, Linux: /dev/ttyUSB0
SERIAL_BAUD = 115200

# Setup Serial Client
try:
    arduino_serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    print(f"[SERIAL] Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"[SERIAL] Warning: Serial connection failed: {e}")
    arduino_serial = None

# Setup MQTT Client
mqtt_publisher = mqtt.Client()
try:
    mqtt_publisher.connect(MQTT_SERVER, MQTT_PORT, 60)
    mqtt_publisher.loop_start()
except Exception as e:
    print(f"[MQTT] Warning: Could not connect to broker: {e}")

# Initialize Head Detection Model
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

# Initialize Face Recognition Model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = "face_model.yml"
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    print("[RECOGNITION] Loaded trained face model.")
else:
    print("[ERROR] Face model not found! Please run enroll.py first.")
    exit()

video_stream = cv2.VideoCapture(0)

# State Tracking
last_published_status = None
last_published_error = None
ERROR_THRESHOLD = 5 # Small threshold for smooth USB tracker 

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
        try:
            mqtt_publisher.publish(HEARTBEAT_TOPIC, json.dumps(payload))
        except:
            pass
        time.sleep(5)

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
        should_publish = False

        if len(detected_faces) > 0:
            # We track the highest confidence match in the frame
            best_match_id = -1
            best_confidence = 1000 # Lower distance is better in LBPH
            tracked_face = None

            for (x_pos, y_pos, w_size, h_size) in detected_faces:
                face_crop = grayscale_frame[y_pos:y_pos+h_size, x_pos:x_pos+w_size]
                id_, distance = face_recognizer.predict(face_crop)
                
                if distance < best_confidence:
                    best_confidence = distance
                    best_match_id = id_
                    tracked_face = (x_pos, y_pos, w_size, h_size)
                    
                # Mark all faces
                cv2.rectangle(current_frame, (x_pos, y_pos), (x_pos+w_size, y_pos+h_size), (0, 0, 255), 2)
            
            # Distance threshold (usually < 70-80 means good match for LBPH)
            if best_match_id == 1 and best_confidence < 85:
                (x_pos, y_pos, w_size, h_size) = tracked_face
                center_x = x_pos + w_size // 2
                current_status, error = determine_movement_command(center_x, current_frame.shape[1])
                detection_certainty = max(0, 1.0 - (best_confidence / 100)) # Approximate to %

                # Draw green bounding box for tracked target
                cv2.rectangle(current_frame, (x_pos, y_pos), (x_pos+w_size, y_pos+h_size), (0, 255, 0), 2)
                cv2.putText(current_frame, f"TARGET LOCKED (Conf: {int(detection_certainty*100)}%)", (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(current_frame, (center_x, y_pos + h_size//2), 4, (0, 255, 0), -1)

        # Logic to "avoid flooding" - only publish when State changes or Error changes significantly
        if current_status != last_published_status:
            should_publish = True
        elif current_status in ["MOVE_LEFT", "MOVE_RIGHT"] and last_published_error is not None:
             if abs(error - last_published_error) > ERROR_THRESHOLD:
                 should_publish = True

        if should_publish:
            # Format and send over serial to Arduino
            if arduino_serial is not None and arduino_serial.is_open:
                try:
                    arduino_serial.write(f"{error}\n".encode())
                except Exception as e:
                    print(f"[SERIAL] Error writing to Arduino: {e}")

            # Send up to MQTT / Dashboard
            msg_payload = {
                "status": current_status,
                "confidence": round(detection_certainty, 2),
                "error": error,
                "timestamp": int(time.time())
            }
            try:
                mqtt_publisher.publish(MOVEMENT_TOPIC, json.dumps(msg_payload))
                print(f"[VISION] Published: {current_status} | Error: {error} | Serial Tx")
            except Exception:
                pass
            
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
    if arduino_serial:
        arduino_serial.close()


