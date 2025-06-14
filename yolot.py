import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load model
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Brighten the frame
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

    # Run detection
    results = model(frame, conf=0.1)

    # Draw boxes and labels
    debug_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Detection", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

