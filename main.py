"""TrafficVision Lite entry point."""

import cv2

from config import CAMERA_SOURCE, SCREENSHOT_DIR, WINDOW_NAME
from detector import VehicleDetector
from ui_overlay import draw_interface
from utils import (
    FPSCounter,
    count_vehicles,
    ensure_directory,
    get_traffic_status,
    save_screenshot,
)


def main():
    ensure_directory(SCREENSHOT_DIR)

    print("TrafficVision Lite")
    print("Loading YOLO model. The first run may download the model file...")

    try:
        detector = VehicleDetector()
    except Exception as error:
        print(f"Could not load YOLO model: {error}")
        return

    camera = cv2.VideoCapture(CAMERA_SOURCE)
    if not camera.isOpened():
        camera.release()
        print("Could not open camera source.")
        print("Check CAMERA_SOURCE in config.py and make sure the camera or stream is available.")
        return

    fps_counter = FPSCounter()
    print("Camera started. Press Q to quit or S to save a screenshot.")

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Could not read a frame from the camera source.")
                break

            try:
                detections = detector.detect(frame)
            except Exception as error:
                print(f"Detection failed: {error}")
                break

            counts = count_vehicles(detections)
            total_vehicles = sum(counts.values())
            traffic_status = get_traffic_status(total_vehicles)
            fps = fps_counter.update()

            display_frame = draw_interface(frame, detections, counts, fps, traffic_status)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                try:
                    screenshot_path = save_screenshot(display_frame, SCREENSHOT_DIR)
                except OSError as error:
                    print(error)
                else:
                    print(f"Screenshot saved: {screenshot_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
