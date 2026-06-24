import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

class TrafficDetector:
    def __init__(self):
        # Load lightweight nano model for Edge performance
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        self.model.to(Config.DEVICE)
        
    def anonymize_frame(self, frame, boxes):
        """Applies data privacy compliance regulations by blurring detections."""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Simulate license plate/face blurring within bounding box lower thirds
            h = y2 - y1
            w = x2 - x1
            roi = frame[y1 + int(h*0.6):y2, x1:x2]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (23, 23), 0)
                frame[y1 + int(h*0.6):y2, x1:x2] = blurred
        return frame

    def process_frame(self, frame):
        """Processes an incoming traffic frame to count vehicles and check overrides."""
        results = self.model(frame, verbose=False)[0]
        
        counts = {cls: 0 for cls in Config.VEHICLE_CLASSES}
        emergency_detected = False
        detected_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            if label in Config.VEHICLE_CLASSES and conf >= Config.CONFIDENCE_THRESHOLD:
                counts[label] = counts.get(label, 0) + 1
                detected_boxes.append(box)
                if label in Config.EMERGENCY_CLASSES:
                    emergency_detected = True

        # Ensure compliance metrics by processing image anonymization
        processed_frame = self.anonymize_frame(frame.copy(), detected_boxes)
        total_vehicles = sum([counts[c] for c in Config.VEHICLE_CLASSES])

        return {
            "total_count": total_vehicles,
            "breakdown": counts,
            "emergency_trigger": emergency_detected,
            "annotated_frame": processed_frame
        }