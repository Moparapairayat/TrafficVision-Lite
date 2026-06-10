"""Premium OpenCV HUD and bounding box drawing functions."""

import cv2

from config import APP_NAME, APP_SUBTITLE


COLORS = {
    "panel": (18, 22, 30),
    "panel_line": (255, 190, 40),
    "text": (245, 247, 250),
    "muted": (168, 178, 190),
    "live": (70, 240, 120),
    "car": (255, 194, 64),
    "motorcycle": (82, 220, 255),
    "bus": (120, 255, 120),
    "truck": (255, 130, 92),
    "heavy": (70, 70, 255),
    "moderate": (70, 190, 255),
    "light": (80, 230, 140),
}


def draw_interface(frame, detections, counts, fps, traffic_status):
    """Draw the complete HUD and all detection boxes on the frame."""
    draw_top_panel(frame, counts, fps, traffic_status)
    draw_bottom_analytics(frame, counts)

    for detection in detections:
        draw_vehicle_box(frame, detection)

    return frame


def draw_top_panel(frame, counts, fps, traffic_status):
    height, width = frame.shape[:2]
    panel_height = 112 if height >= 500 else 96

    draw_transparent_rect(frame, (0, 0), (width, panel_height), COLORS["panel"], 0.72)
    cv2.line(frame, (0, panel_height - 1), (width, panel_height - 1), COLORS["panel_line"], 2)

    cv2.putText(
        frame,
        APP_NAME,
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        COLORS["text"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        APP_SUBTITLE,
        (24, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLORS["muted"],
        1,
        cv2.LINE_AA,
    )

    draw_live_indicator(frame, width)

    total = sum(counts.values())
    metric_y = 72 if panel_height >= 112 else 66
    status_color = traffic_status_color(traffic_status)
    available_width = max(260, width - 44)
    gap = 8
    fps_width = 96 if width >= 560 else 82
    vehicles_width = 118 if width >= 560 else 92
    status_width = max(132, available_width - fps_width - vehicles_width - gap * 2)
    status_text = traffic_status if status_width >= 170 else traffic_status.replace(" Traffic", "")

    metrics = [
        ("FPS", f"{fps:.1f}", COLORS["motorcycle"], fps_width),
        ("COUNT", str(total), COLORS["car"], vehicles_width),
        ("STATUS", status_text, status_color, status_width),
    ]

    x = 22
    for label, value, color, box_width in metrics:
        draw_metric_chip(frame, x, metric_y, box_width, 28, label, value, color)
        x += box_width + 10


def draw_live_indicator(frame, width):
    x = max(22, width - 104)
    y = 31
    cv2.circle(frame, (x, y - 6), 6, COLORS["live"], -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "LIVE",
        (x + 14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLORS["text"],
        1,
        cv2.LINE_AA,
    )


def draw_metric_chip(frame, x, y, width, height, label, value, color):
    draw_transparent_rect(frame, (x, y), (x + width, y + height), (32, 38, 48), 0.62)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x + 8, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLORS["muted"],
        1,
        cv2.LINE_AA,
    )
    value_x = x + 72 if width >= 112 else x + 48

    cv2.putText(
        frame,
        value,
        (value_x, y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLORS["text"],
        1,
        cv2.LINE_AA,
    )


def draw_bottom_analytics(frame, counts):
    height, width = frame.shape[:2]
    bar_height = 72 if height >= 500 else 60
    y1 = height - bar_height
    margin = 18

    draw_transparent_rect(frame, (0, y1), (width, height), COLORS["panel"], 0.68)
    cv2.line(frame, (0, y1), (width, y1), COLORS["panel_line"], 2)

    items = [
        ("Cars/Jeeps", counts.get("car", 0), COLORS["car"]),
        ("Motorcycles", counts.get("motorcycle", 0), COLORS["motorcycle"]),
        ("Buses", counts.get("bus", 0), COLORS["bus"]),
        ("Trucks", counts.get("truck", 0), COLORS["truck"]),
    ]

    segment_width = max(1, (width - margin * 2) // len(items))
    label_scale = 0.43 if width >= 620 else 0.34

    for index, (label, value, color) in enumerate(items):
        x = margin + index * segment_width
        if index > 0:
            cv2.line(frame, (x - 8, y1 + 14), (x - 8, height - 14), (70, 78, 90), 1)

        cv2.putText(
            frame,
            label,
            (x, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            COLORS["muted"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(value),
            (x, y1 + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.88,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_vehicle_box(frame, detection):
    x1, y1, x2, y2 = clamp_box(detection["bbox"], frame.shape)
    class_name = detection["class_name"]
    color = COLORS.get(class_name, COLORS["panel_line"])

    draw_corner_box(frame, x1, y1, x2, y2, color)
    draw_detection_label(frame, x1, y1, detection, color)


def draw_corner_box(frame, x1, y1, x2, y2, color):
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    corner = min(38, max(18, int(min(box_width, box_height) * 0.24)))
    thickness = 2

    # A faint full outline keeps the object readable while the corners feel modern.
    cv2.rectangle(frame, (x1, y1), (x2, y2), darken(color), 1, cv2.LINE_AA)

    lines = [
        ((x1, y1), (x1 + corner, y1)),
        ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)),
        ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)),
        ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)),
        ((x2, y2), (x2, y2 - corner)),
    ]

    for start, end in lines:
        cv2.line(frame, start, end, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_detection_label(frame, x1, y1, detection, color):
    label = f"{detection['display_name']} {detection['confidence'] * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    text_width, text_height = text_size
    padding_x = 8
    padding_y = 6

    label_x1 = x1
    label_y2 = y1 - 6
    label_y1 = label_y2 - text_height - padding_y * 2

    if label_y1 < 0:
        label_y1 = y1 + 6
        label_y2 = label_y1 + text_height + padding_y * 2

    label_x2 = label_x1 + text_width + padding_x * 2
    frame_height, frame_width = frame.shape[:2]
    if label_x2 > frame_width - 1:
        shift = label_x2 - frame_width + 1
        label_x1 = max(0, label_x1 - shift)
        label_x2 = frame_width - 1

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (13, 18, 26), -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, 1, cv2.LINE_AA)
    cv2.line(frame, (label_x1, label_y2), (label_x2, label_y2), color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (label_x1 + padding_x, label_y2 - padding_y - baseline // 2),
        font,
        scale,
        COLORS["text"],
        thickness,
        cv2.LINE_AA,
    )


def draw_transparent_rect(frame, top_left, bottom_right, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def traffic_status_color(status):
    if status == "Heavy Traffic":
        return COLORS["heavy"]
    if status == "Moderate Traffic":
        return COLORS["moderate"]
    return COLORS["light"]


def darken(color, factor=0.45):
    return tuple(int(channel * factor) for channel in color)


def clamp_box(bbox, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2
