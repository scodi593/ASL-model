import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model # type: ignore

# Configuration
MODEL_PATH = "asl_model.h5"
CLASS_INDICES = "class_indices.json"
IMG_SIZE = (128, 128)
ROI_SIZE = 200  # Size of detection window

# Load model and class indices
model = load_model(MODEL_PATH)
with open(CLASS_INDICES, 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame and convert to RGB
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Create ROI rectangle
    start_x = width//2 - ROI_SIZE//2
    start_y = height//2 - ROI_SIZE//2
    end_x = start_x + ROI_SIZE
    end_y = start_y + ROI_SIZE
    
    # Draw ROI rectangle
    cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Extract and preprocess ROI
    roi = frame[start_y:end_y, start_x:end_x]
    roi = cv2.resize(roi, IMG_SIZE)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)
    
    # Make prediction
    prediction = model.predict(roi)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display prediction
    text = f"{predicted_class} ({confidence:.2f}%)"
    cv2.putText(display_frame, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("ASL Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()