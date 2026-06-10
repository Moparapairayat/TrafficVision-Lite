"""YOLO vehicle detection logic."""

from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, MODEL_PATH, VEHICLE_CLASSES


class VehicleDetector:
    """Small wrapper around Ultralytics YOLO for vehicle-only detection."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        vehicle_classes=VEHICLE_CLASSES,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = set(vehicle_classes)
        self.class_names = self.model.names

    def detect(self, frame):
        """Return clean vehicle detections for one OpenCV frame."""
        if frame is None:
            return []

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=sorted(self.vehicle_classes),
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            class_id = int(box.cls.item())
            if class_id not in self.vehicle_classes:
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            class_name = self._class_name(class_id)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "display_name": self._display_name(class_name),
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections

    def _class_name(self, class_id):
        if isinstance(self.class_names, dict):
            return self.class_names.get(class_id) or self.class_names.get(str(class_id), str(class_id))

        try:
            return self.class_names[class_id]
        except (IndexError, KeyError, TypeError):
            return str(class_id)

    @staticmethod
    def _display_name(class_name):
        if class_name == "car":
            return "Car/Jeep"
        return class_name.replace("_", " ").title()
