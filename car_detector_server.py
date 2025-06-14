import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import threading

app = Flask(__name__)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
car_detected = False

# HTML page
HTML_PAGE = """
<!doctype html>
<title>Driveway Monitor</title>
<h1>Live Driveway Feed</h1>
<p>Status: <b style="color:{{ 'green' if detected else 'red' }}">{{ '🚗 Car Detected!' if detected else 'No Car Detected' }}</b></p>
<img src="{{ url_for('video') }}">
"""

def detect_car(frame):
    global car_detected
    results = model(frame, verbose=False)
    for r in results:
        for cls in r.boxes.cls:
            if int(cls) == 2:  # Class 2 = 'car' in COCO
                car_detected = True
                return
    car_detected = False

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run car detection in a separate thread
        threading.Thread(target=detect_car, args=(frame.copy(),)).start()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, detected=car_detected)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

