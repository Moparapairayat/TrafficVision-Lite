"""Utility helpers for TrafficVision Lite."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2


def ensure_directory(path):
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


class FPSCounter:
    """Simple smoothed FPS calculator."""

    def __init__(self, smoothing=0.9):
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be greater than or equal to 0 and less than 1")

        self.smoothing = smoothing
        self.previous_time = perf_counter()
        self.fps = 0.0

    def update(self):
        current_time = perf_counter()
        elapsed = current_time - self.previous_time
        self.previous_time = current_time

        if elapsed <= 0:
            return self.fps

        instant_fps = 1.0 / elapsed
        if self.fps == 0.0:
            self.fps = instant_fps
        else:
            self.fps = (self.fps * self.smoothing) + (instant_fps * (1 - self.smoothing))

        return self.fps


def save_screenshot(frame, screenshot_dir):
    """Save the current displayed frame with a timestamped filename."""
    if frame is None:
        raise OSError("Could not save screenshot: frame is empty.")

    ensure_directory(screenshot_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = Path(screenshot_dir) / f"trafficvision_{timestamp}.jpg"

    try:
        saved = cv2.imwrite(str(file_path), frame)
    except cv2.error as error:
        raise OSError(f"Could not save screenshot: {error}") from error

    if not saved:
        raise OSError(f"Could not save screenshot: {file_path}")

    return file_path


def get_traffic_status(total_vehicles):
    """Return a friendly traffic status based on the total vehicle count."""
    try:
        total_vehicles = max(0, int(total_vehicles))
    except (TypeError, ValueError):
        total_vehicles = 0

    if total_vehicles <= 3:
        return "Light Traffic"
    if total_vehicles <= 8:
        return "Moderate Traffic"
    return "Heavy Traffic"


def count_vehicles(detections):
    """Count detections by vehicle class."""
    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

    for detection in detections or []:
        if not isinstance(detection, dict):
            continue

        class_name = detection.get("class_name")
        if class_name in counts:
            counts[class_name] += 1

    return counts
