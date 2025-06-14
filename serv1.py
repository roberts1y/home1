import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

model = YOLO("yolov8n.pt")  # Load Nano model
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

car_detected = False
last_alert_time = 0
CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

HTML_PAGE = """
<!doctype html>
<title>Driveway Monitor</title>
<h1>Live Driveway Feed</h1>
<p>Status: <b style="color:{{ 'green' if detected else 'red' }}">{{ '🚗 Car Detected!' if detected else 'No Car Detected' }}</b></p>
<img src="{{ url_for('video') }}">
"""

def detect_car_and_display(frame):
    global car_detected
    # Brighten the frame for night/rain
    bright = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
    results = model(bright, conf=0.1)

    # Visualize detections
    debug_frame = results[0].plot()
    cv2.imshow("YOLO Detection", debug_frame)
    cv2.waitKey(1)

    # Check if any target class is detected
    found = False
    for r in results:
        for cls in r.boxes.cls:
            if int(cls) in CAR_CLASSES:
                found = True
                break
    car_detected = found

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run detection in background thread
        threading.Thread(target=detect_car_and_display, args=(frame.copy(),)).start()

        # Stream raw frame to browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, detected=car_detected)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting server... Visit http://localhost:5000 or http://<your-ip>:5000")
    app.run(host="0.0.0.0", port=5000)

