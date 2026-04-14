import time
from utils.logger import logger

class BehaviorAnalyzer:
    def __init__(self, side_look_threshold=30, side_look_count=3, time_window=30):
        self.side_look_threshold = side_look_threshold
        self.side_look_count = side_look_count
        self.time_window = time_window
        self.history = {}  # {id: [{timestamp, gaze, bbox, ...}]}
        self.last_flagged = {} # {id: {type: timestamp}}
        
    def _should_flag(self, tid, etype, current_time):
        if tid not in self.last_flagged:
            self.last_flagged[tid] = {}
        last_t = self.last_flagged[tid].get(etype, 0)
        if current_time - last_t > 3.0: # 3 second cooldown
            self.last_flagged[tid][etype] = current_time
            return True
        return False
        
    def analyze(self, tracks, gaze_data, detections):
        current_time = time.time()
        suspicious_events = []

        track_ids = list(tracks.keys())
        for i, tid in enumerate(track_ids):
            if tid not in self.history:
                self.history[tid] = []
            
            tdata = tracks[tid]
            bbox = tdata["bbox"]
            
            event = {
                "timestamp": current_time,
                "gaze": gaze_data.get(tid, "center"),
                "bbox": bbox,
                "has_phone": self._check_object_proximity(bbox, detections, 67),
                "has_chit": self._check_object_proximity(bbox, detections, 73),
                "has_watch": self._check_object_proximity(bbox, detections, 74)
            }
            self.history[tid].append(event)
            
            # Clean old history (keep last 30 seconds)
            self.history[tid] = [e for e in self.history[tid] if current_time - e["timestamp"] < self.time_window]
            
            # 1. Detect Prohibited Objects (instant detection)
            if event["has_phone"] and self._should_flag(tid, "phone", current_time):
                suspicious_events.append({"id": tid, "type": "phone", "description": "Using a mobile phone"})
                logger.info(f"ALERT: ID {tid} - phone detected!")
            if event["has_watch"] and self._should_flag(tid, "smartwatch", current_time):
                suspicious_events.append({"id": tid, "type": "smartwatch", "description": "Looking at smartwatch"})
                logger.info(f"ALERT: ID {tid} - smartwatch detected!")
            if event["has_chit"] and self._should_flag(tid, "chit", current_time):
                suspicious_events.append({"id": tid, "type": "chit", "description": "Suspicious object/chit detected"})
                logger.info(f"ALERT: ID {tid} - chit detected!")
            
            # 2. Side Looking Detection (require 5 instances in last 10 seconds)
            recent_10s = [e for e in self.history[tid] if current_time - e["timestamp"] < 10.0]
            side_looks = [e for e in recent_10s if e["gaze"] in ["left", "right"]]
            
            if len(side_looks) >= 5:  # 5 side-looks in 10 seconds = suspicious
                if self._should_flag(tid, "side_look", current_time):
                    suspicious_events.append({"id": tid, "type": "side_look", "description": "Frequent side-looking"})
                    logger.info(f"SUSPICIOUS: ID {tid} - frequent side-looking ({len(side_looks)} times in 10s)")
            
            # 3. Proximity Detection - ONLY flag if VERY close AND sustained
            for other_tid in track_ids[i+1:]:
                other_bbox = tracks[other_tid]["bbox"]
                h_dist = self._calculate_horizontal_proximity(bbox, other_bbox)
                
                # Only flag if side-by-side bounding boxes physically intersect (horizontal gap == 0px)
                if h_dist == 0:
                    if self._should_flag(tid, "talking", current_time):
                        suspicious_events.append({"id": tid, "type": "talking", "description": "Exchanging papers / Talking"})
                        logger.info(f"SUSPICIOUS: ID {tid} <-> ID {other_tid} - exchanging (gap: {int(h_dist)}px)")
                    if self._should_flag(other_tid, "talking", current_time):
                        suspicious_events.append({"id": other_tid, "type": "talking", "description": "Exchanging papers / Talking"})
                
        return suspicious_events

    def _check_object_proximity(self, person_bbox, detections, cls_id):
        """Checks if a prohibited object is within or very near a person's bounding box."""
        for det in detections:
            if det["class"] == cls_id:
                obj_bbox = det["bbox"]
                if self._calculate_bbox_overlap(person_bbox, obj_bbox) > 0:
                    return True
        return False

    def _calculate_horizontal_proximity(self, b1, b2):
        cy1 = (b1[1] + b1[3]) / 2
        cy2 = (b2[1] + b2[3]) / 2
        if abs(cy1 - cy2) > 80:
            return float('inf')
        dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        return dx

    def _calculate_bbox_overlap(self, b1, b2):
        """Simple intersection check."""
        dx = min(b1[2], b2[2]) - max(b1[0], b2[0])
        dy = min(b1[3], b2[3]) - max(b1[1], b2[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0
