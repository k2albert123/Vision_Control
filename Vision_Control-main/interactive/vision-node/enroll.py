import cv2
import os
import numpy as np

def enroll_face():
    # Initialize Head Detection Model
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(cascade_path)

    video_stream = cv2.VideoCapture(0)

    print("[ENROLLMENT] Look at the camera. Capturing 50 samples...")
    
    face_samples = []
    ids = []
    sample_count = 0
    max_samples = 50

    while True:
        success, current_frame = video_stream.read()
        if not success:
            print("[ERROR] Could not read from webcam.")
            break
            
        grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(grayscale_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in detected_faces:
            sample_count += 1
            face_crop = grayscale_frame[y:y+h, x:x+w]
            
            face_samples.append(face_crop)
            ids.append(1) # We use ID 1 for the single enrolled user

            # Draw visual feedback
            cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(current_frame, f"Capturing: {sample_count}/{max_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show the frame
            cv2.imshow("Enrolling Face", current_frame)
            cv2.waitKey(100) # Wait 100ms between captures

        cv2.imshow("Enrolling Face", current_frame)
        
        if sample_count >= max_samples:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()

    if len(face_samples) > 0:
        print(f"\n[TRAINING] Training LBPH model on {len(face_samples)} face samples...")
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(face_samples, np.array(ids))
            recognizer.write("face_model.yml")
            print("[SUCCESS] Model trained and saved as 'face_model.yml'")
            print("You can now run vision_node.py to track this specific face!")
        except AttributeError:
             print("\n[ERROR] cv2.face module not found!")
             print("Please ensure you run: pip install opencv-contrib-python")
    else:
        print("[ERROR] No face samples captured. Enrollment failed.")

if __name__ == "__main__":
    enroll_face()
