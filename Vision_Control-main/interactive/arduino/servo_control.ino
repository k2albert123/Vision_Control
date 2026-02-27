#include <Servo.h>

Servo trackerServo;
int currentPos = 90; // Start centered

// Proportional tracking constants
const float Kp = 0.05;
const int MAX_STEP = 5;

void setup() {
  Serial.begin(115200);
  trackerServo.attach(9);
  updateServo(currentPos);
  Serial.println("Arduino Tracker Ready");
}

void loop() {
  // Check if we received an error value over USB Serial
  if (Serial.available() > 0) {
    // Read the tracking error (in pixels) from the PC camera
    int error = Serial.parseInt();
    
    // Clear any trailing newlines or characters
    while(Serial.available() > 0) {
      Serial.read();
    }

    // Process movement if the error is non-zero
    if (error != 0) {
      int stepSize = min(MAX_STEP, abs(error) * Kp);
      
      if (error < 0) {
        currentPos -= stepSize; // Move Left
      } else {
        currentPos += stepSize; // Move Right
      }
      
      currentPos = constrain(currentPos, 0, 180);
      updateServo(currentPos);
    }
  }
}

void updateServo(int degrees) {
  // Map our virtual 0-180 to the physical pulse widths or valid servo range if needed
  // For standard servos, direct write is usually fine.
  trackerServo.write(degrees);
}
