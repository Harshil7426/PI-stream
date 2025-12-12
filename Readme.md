# Live PI stream with human detection
## Description of project

Smart Drone Cam Dashboard is a real-time web dashboard that captures video from a camera (Raspberry Pi camera, ESP32-CAM, or USB webcam), processes each video frame with OpenCV, optionally runs YOLOv8 human detection, applies vision filters (thermal, edge, night), and streams the processed video over the local network using Flask (MJPEG). The dashboard provides sliders to adjust brightness, contrast and saturation live, and includes an experimental tracking branch (SORT) to keep consistent IDs for detected people.

---

## What tech is used

* **Python 3.8+**
* **Flask** — web server + MJPEG streaming
* **OpenCV (cv2)** — frame capture & image processing
* **Ultralytics / YOLOv8** — AI human detection (yolov8n recommended)
* **NumPy** — numerical operations
* **(Optional) SORT / filterpy** — Lightweight tracking (testing branch)
* **MJPEG stream** — low-latency multipart streaming to browser
* Frontend: **HTML5 / CSS3 / JavaScript** (templates/index.html)

---

## Flow of the project (simple ASCII diagram)

```
[Camera (PiCam / ESP32-CAM / USB Webcam)]
             │
             │ HTTP MJPEG / USB / RTSP
             ▼
       [Flask App: cam.py / testing.py]
  ┌─────────────────────────────────────────┐
  │ 1. cv2.VideoCapture(stream_url or 0)    │
  │ 2. Read frame                           │
  │ 3. Apply Brightness/Contrast/Saturation │
  │ 4. Apply vision mode (Normal/YOLO/Therm/Edge/Night)
  │ 5. (Optional) YOLOv8 inference → boxes  │
  │ 6. (Optional) SORT tracker assigns IDs  │
  │ 7. Encode JPEG → yield multipart frame  │
  └─────────────────────────────────────────┘
             │
             ▼
[Device on same LAN (browser)] → Connect to http://<host>:<port> and view dashboard
```

### Typical use-case described in words

1. Camera (Raspberry Pi or ESP32-CAM) streams video over the local network (or a USB webcam is attached to the host).
2. `cam.py` opens that stream (via OpenCV).
3. The app processes and augments frames (filters, AI detection).
4. Flask serves the processed frames as an MJPEG stream and a frontend UI.
5. Any device on the same network opens the dashboard URL (e.g., `http://192.168.0.120:5001`) and sees live processed video with controls.

---

## What options the project provides (features)

* **Live Video Streaming** (MJPEG, low latency)
* **Human Detection (YOLOv8)** — bounding boxes + confidence
* **Vision Modes**

  * Normal (raw RGB)
  * YOLO (detection overlays)
  * Thermal (simulated heatmap using `COLORMAP_JET`)
  * Edge (Canny edge detector)
  * Night (simulated low-light color map)
* **Image Controls** — live sliders for Brightness, Contrast, Saturation
* **Tracking (testing.py)** — experimental SORT-based tracker, assigns persistent IDs
* **Configurable stream source** — IP camera URL (e.g., Pi/ESP32), or `0` for local webcam
* **Adjustable FPS / detection frequency** — reduce inference frequency to save CPU

---

## Packages required

Copy-paste the install command below (recommended inside a virtualenv):

```bash
# create and activate venv (optional but recommended)
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install --upgrade pip
pip install flask opencv-python ultralytics numpy
# Optional (for SORT tracking):
pip install filterpy
```

> **Notes**
>
> * If `opencv-python` causes issues (on Raspberry Pi), consider `opencv-python-headless` or building OpenCV from source with required options.
> * `ultralytics` will pull the YOLOv8 model loader; keep `yolov8n.pt` in the project root (or let ultralytics download it).

---

## Before you start — prepare the camera stream

Choose one of these options depending on your hardware:

### Option A — Raspberry Pi (recommended)

1. Install `mjpg-streamer` (or use `raspivid` + `netcat` for older setups).
2. Example `mjpg-streamer` command for PiCam:

```bash
# run from mjpg-streamer folder
./mjpg_streamer -i "./input_raspicam.so -fps 20 -q 70" -o "./output_http.so -w ./www -p 8080"
```

3. Stream URL to use in `cam.py`:

```
http://<PI-IP>:8080/?action=stream
```

### Option B — ESP32-CAM

1. Flash the `CameraWebServer` example via Arduino or PlatformIO.
2. Find the ESP32 IP in the serial monitor.
3. Typical stream URL:

```
http://<ESP32-IP>:81/stream
```

### Option C — USB webcam (local machine)

No network setup needed. Use `cv2.VideoCapture(0)` in `cam.py`.

---

## How to run the project — step by step (detailed)

### 1. Clone repository and prepare environment

```bash
git clone https://github.com/<your-repo>/smart-drone-cam.git
cd smart-drone-cam

# create + activate venv (optional but recommended)
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install flask opencv-python ultralytics numpy
# Optional: for tracking in testing.py
pip install filterpy
```

### 2. Place the model file

* Put `yolov8n.pt` in the project root (same folder as `cam.py`).
* Or update `cam.py` to load the model via ultralytics API (it can download automatically).

### 3. Configure `cam.py` (open the file and edit)

Locate the top section where stream is defined. Examples:

```python
# Use IP camera stream (Pi or ESP32)
stream_url = "http://192.168.0.110:8080/?action=stream"
cap = cv2.VideoCapture(stream_url)

# OR use local webcam
# cap = cv2.VideoCapture(0)
```

Optional parameters to adjust:

* `fps_limit` — cap FPS to reduce CPU
* `DETECT_EVERY` — (in testing.py) run detection every N frames
* `HOST` and `PORT` for Flask run: `app.run(host="0.0.0.0", port=5001)`

### 4. Start the backend

**Main dashboard (detection + filters):**

```bash
python cam.py
```

**Experimental tracking branch (YOLO + SORT):**

```bash
python testing.py
```

You should see Flask console logs. If using `host="0.0.0.0"`, the app is reachable by other devices on the LAN.

### 5. Open the dashboard (on any device on the same LAN)

In a browser enter:

```
http://<HOST-IP>:5001
# Example: http://192.168.0.120:5001
```

You will see:

* Live video feed
* Buttons or dropdown for Vision Mode (Normal / YOLO / Thermal / Edge / Night)
* Sliders for Brightness / Contrast / Saturation
* (If testing.py) tracking IDs displayed

---

## Raspberry Pi quick start example (compact)

1. Install mjpg-streamer and start stream on Pi (port 8080).
2. On your desktop/laptop:

   * Clone repo, create venv, install deps.
   * Edit `cam.py` → set `stream_url = "http://<PI-IP>:8080/?action=stream"`.
   * Run `python cam.py`.
3. Open `http://<your-desktop-ip>:5001` or run Flask with `0.0.0.0` and open `http://<desktop-ip>:5001` from phone/tablet.

---

## Configuration tips & recommended settings

* On Raspberry Pi, set `DETECT_EVERY = 3` or more to limit heavy inference.
* Reduce camera resolution (e.g., 640×480) for better FPS on low-power hardware.
* Use `yolov8n.pt` (Nano) for edge devices; `yolov8s` or larger for servers.
* If `opencv-python` fails on Pi, try `opencv-python-headless` or install via apt.

---

## Troubleshooting

* **No video in browser**: verify `stream_url` is reachable directly first (open it in browser). Check firewall/router settings.
* **Model load errors**: confirm `yolov8n.pt` path and that `ultralytics` is installed.
* **High CPU / low FPS**: lower detection frequency or resolution; consider using a Coral / NPU or a stronger host.
* **Blank frames or error reading camera**: try `cv2.VideoCapture(0)` for USB webcam test; check `/dev/video0` permissions on Linux.

---

## Files you will interact with

* `cam.py` — main app: streaming, filters, YOLO inference
* `testing.py` — experimental tracking: YOLO + SORT
* `yolov8n.pt` — model weights (place in project root)
* `templates/index.html` — frontend dashboard
* `static/` — optional CSS/JS assets

---

## Example useful commands (summary)

```bash
# clone + venv
git clone https://github.com/<you>/smart-drone-cam.git
cd smart-drone-cam
python -m venv venv
source venv/bin/activate

# install
pip install flask opencv-python ultralytics numpy
pip install filterpy     # optional

# run
python cam.py            # main dashboard
python testing.py        # experimental tracker
```

---

