import cv2
from datetime import datetime
import os

# Save directory — use full path
output_dir = "/home/rob/Documents/projects/home1/captured_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Camera not found.")
    exit(1)

# Capture a single frame
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture image.")
    exit(1)

# Save the image with a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(output_dir, f"{timestamp}.jpg")
cv2.imwrite(filename, frame)

print(f"✅ Image saved: {filename}")

