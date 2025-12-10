from flask import Flask, render_template, Response, request
import cv2, time
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# ===============================
#       GLOBAL VARIABLES
# ===============================
model = YOLO("yolov8n.pt")   # YOLO loaded once
stream_url = "http://192.168.0.110:8080/?action=stream"
cap = cv2.VideoCapture(stream_url)

current_mode = "normal"

brightness = 1.0
contrast = 1.0
saturation = 1.0
fps_limit = 0   # <-- 0 means NO LIMIT (max FPS)


# ===============================
#       FRAME PROCESSING
# ===============================
def apply_color_controls(frame):
    global brightness, contrast, saturation

    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=(brightness - 1) * 50)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] *= saturation
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame


def process_frame(frame, mode):
    frame = apply_color_controls(frame)

    if mode == "yolo":
        results = model.predict(source=frame, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Human", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif mode == "thermal":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    elif mode == "edge":
        edges = cv2.Canny(frame, 60, 150)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif mode == "night":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

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

        cv2.putText(frame, f"{current_mode.upper()} | FPS:{fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        # ---- NO FPS THROTTLE ----
        # (fps_limit = 0 means unlimited)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
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
