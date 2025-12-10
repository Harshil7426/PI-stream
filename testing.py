from flask import Flask, render_template, Response
import cv2, time
from ultralytics import YOLO
import numpy as np
from sort import Sort   # SORT Tracker

app = Flask(__name__)

# ===============================
#       GLOBAL VARIABLES
# ===============================
model = YOLO("yolov8n.pt")
stream_url = "http://192.168.0.110:8080/?action=stream"
cap = cv2.VideoCapture(stream_url)

current_mode = "normal"

brightness = 1.0
contrast = 1.0
saturation = 1.0

# YOLO only every N frames
DETECT_EVERY = 3
frame_counter = 0

# Initialize SORT Tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

# ===============================
#   COLOR CONTROLS
# ===============================
def apply_color_controls(frame):
    global brightness, contrast, saturation

    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=(brightness - 1) * 50)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] *= saturation
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ===============================
#   PROCESS FRAME (YOLO + SORT)
# ===============================
def process_frame(frame, mode):
    global frame_counter, tracker

    frame = apply_color_controls(frame)

    if mode != "yolo":
        return frame

    frame_counter += 1
    detections = []

    # 🔥 YOLO every 3rd frame only
    if frame_counter % DETECT_EVERY == 0:
        results = model.predict(frame, imgsz=320, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2, 0.9])

    # Convert to numpy array for SORT
    detections_np = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    # 🔥 SORT updates tracking
    trackers = tracker.update(detections_np)

    # Draw tracked boxes
    for d in trackers:
        x1, y1, x2, y2, obj_id = map(int, d)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return frame


# ===============================
#       MJPEG STREAM GENERATOR
# ===============================
def gen_frames():
    prev = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = process_frame(frame, current_mode)

        fps = 1 / (time.time() - prev)
        prev = time.time()

        cv2.putText(frame, f"{current_mode.upper()} | FPS:{fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ===============================
#       FLASK ROUTES
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    current_mode = mode
    return ("", 204)

@app.route("/set_param/<param>/<value>")
def set_param(param, value):
    global brightness, contrast, saturation

    v = float(value)
    if param == "brightness":
        brightness = v
    elif param == "contrast":
        contrast = v
    elif param == "saturation":
        saturation = v

    return ("", 204)


# ===============================
#       START SERVER
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
