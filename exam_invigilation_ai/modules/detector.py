from ultralytics import YOLO
import cv2
from utils.logger import logger

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, target_classes=[0, 67, 73, 74]):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        logger.info(f"YOLOv8 Detector initialized with model {model_path}")

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, classes=self.target_classes, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "class": cls,
                    "confidence": conf,
                    "bbox": xyxy
                })
        
        return detections
