# TrafficVision Lite

Real-time vehicle detection using Python, OpenCV, and Ultralytics YOLO.

This project opens a camera feed, detects vehicles, and shows simple traffic information on top of the video. It is mainly built as a small computer vision project for learning and testing.

## Author

MOPARA PAIR AYAT

## Features

- Live camera feed with OpenCV
- Vehicle detection with Ultralytics YOLO
- Detects cars, motorcycles, buses, and trucks
- Shows `Car/Jeep` for car detections
- Vehicle labels with confidence percentage
- Total vehicle count
- Class-wise vehicle count
- FPS display
- Basic traffic status:
  - 0 to 3 vehicles: Light Traffic
  - 4 to 8 vehicles: Moderate Traffic
  - 9 or more vehicles: Heavy Traffic
- Screenshot capture from the running video

## Project Files

```text
trafficvision-lite/
|
|- main.py
|- detector.py
|- ui_overlay.py
|- config.py
|- utils.py
|- requirements.txt
|- .gitignore
|- README.md
|
`- output/
   `- screenshots/
```

## Setup

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

On the first run, Ultralytics may download the YOLO model file from the value set in `config.py`.

## Camera Settings

The default camera source is set in `config.py`:

```python
CAMERA_SOURCE = 0
```

If you have more than one webcam, try `1` or `2`.

For a phone IP camera, use the stream URL from your camera app:

```python
CAMERA_SOURCE = "http://192.168.1.10:8080/video"
```

For CCTV or DVR/NVR RTSP streams, update the same setting:

```python
CAMERA_SOURCE = "rtsp://username:password@192.168.1.50:554/stream1"
```

The exact URL depends on the camera or DVR/NVR model.

## Controls

- `Q`: quit
- `S`: save screenshot to `output/screenshots/`

## Main Settings

Most project settings are in `config.py`:

```python
CAMERA_SOURCE = 0
MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]
SCREENSHOT_DIR = "output/screenshots"
```

## Planned Improvements

- Line crossing counter
- Vehicle tracking IDs
- Basic speed estimation
- Better traffic reports

## Notes

This project does not include number plate recognition, a database, or a web dashboard. It is focused on live detection and a clean OpenCV display.

Generated files such as Python cache files, virtual environments, local `.env` files, YOLO weight files, and saved screenshots are ignored by Git.
