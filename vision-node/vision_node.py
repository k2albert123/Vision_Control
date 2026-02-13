import cv2
import paho.mqtt.client as mqtt
import json
import time

# ========== CONFIG PARAMETERS ==========
GROUP_NAME = "TeAmOnE"
MQTT_SERVER = "10.12.73.101"
MQTT_PORT = 1883

PUBLISH_TOPIC = f"vision/{GROUP_NAME}/movement"

# Setup MQTT Client
mqtt_publisher = mqtt.Client()
mqtt_publisher.connect(MQTT_SERVER, MQTT_PORT, 60)

# Initialize Head Detection Model
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

video_stream = cv2.VideoCapture(0)

def determine_movement_command(face_x_pos, width_of_frame):
    midpoint = width_of_frame // 2
    dead_zone = 50

    if face_x_pos < midpoint - dead_zone:
        return "MOVE_LEFT"
    elif face_x_pos > midpoint + dead_zone:
        return "MOVE_RIGHT"
    else:
        return "CENTERED"

while True:
    success, current_frame = video_stream.read()
    if not success:
        break
        
    grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(grayscale_frame, 1.3, 5)

    current_status = "NO_FACE"
    detection_certainty = 0.0

    if len(detected_faces) > 0:
        (x_pos, y_pos, w_size, h_size) = detected_faces[0]
        center_x = x_pos + w_size // 2
        current_status = determine_movement_command(center_x, current_frame.shape[1])
        detection_certainty = 0.9

        # Draw bounding box
        cv2.rectangle(current_frame, (x_pos, y_pos), (x_pos+w_size, y_pos+h_size), (0, 255, 0), 2)

    msg_payload = {
        "status": current_status,
        "confidence": detection_certainty,
        "timestamp": int(time.time())
    }

    mqtt_publisher.publish(PUBLISH_TOPIC, json.dumps(msg_payload))

    cv2.imshow("Surveillance Feed", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
