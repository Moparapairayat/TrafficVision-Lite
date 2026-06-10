"""Easy-to-edit settings for TrafficVision Lite."""

# Camera source examples:
# 0                         -> default webcam
# "http://192.168.1.10:8080/video" -> phone IP camera URL
# "rtsp://user:pass@ip:554/stream1" -> CCTV RTSP stream
CAMERA_SOURCE = 0

# Ultralytics will download this model automatically on first run if needed.
MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.4

# COCO class IDs used by YOLO:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

SCREENSHOT_DIR = "output/screenshots"

APP_NAME = "TrafficVision Lite"
APP_SUBTITLE = "AI-Powered Real-Time Vehicle Detection System"
WINDOW_NAME = "TrafficVision Lite"
