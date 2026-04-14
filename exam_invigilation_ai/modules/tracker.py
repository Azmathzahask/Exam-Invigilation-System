from utils.logger import logger

class Tracker:
    def __init__(self, algorithm="bytetrack", max_age=999999):
        self.algorithm = algorithm
        self.tracks = {}  # {id: {"bbox": [], "age": 0}}
        self.next_id = 0
        self.max_age = max_age # Frames to keep a lost track (virtually infinite for exams)
        logger.info(f"Tracker initialized using {algorithm}")

    def update(self, detections, frame):
        # Filter for persons (class 0)
        person_detections = [d for d in detections if d["class"] == 0]
        
        # Increment age for all existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            
        new_track_assignments = {}
        used_det_indices = set()
        
        # 1. Try to match with IOU first
        for tid, tdata in self.tracks.items():
            best_iou = 0.5
            best_det_idx = -1
            
            for i, det in enumerate(person_detections):
                if i in used_det_indices: continue
                
                iou = self._calculate_iou(tdata["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            # Fallback 1.5: If IoU fails, match by closest geometric center (Seat coordinate)
            # This ensures if a student stands up drastically changing IoU, they keep their ID
            if best_det_idx == -1:
                best_dist = 100 # Maximum pixel distance to reclaim ID
                old_box = tdata["bbox"]
                cx1 = (old_box[0] + old_box[2]) / 2
                cy1 = (old_box[1] + old_box[3]) / 2
                
                for i, det in enumerate(person_detections):
                    if i in used_det_indices: continue
                    new_box = det["bbox"]
                    cx2 = (new_box[0] + new_box[2]) / 2
                    cy2 = (new_box[1] + new_box[3]) / 2
                    
                    dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = i
            
            if best_det_idx != -1:
                new_track_assignments[tid] = person_detections[best_det_idx]["bbox"]
                used_det_indices.add(best_det_idx)
                self.tracks[tid]["age"] = 0 # Reset age on hit

        # 2. Assign new IDs for unmatched detections
        for i, det in enumerate(person_detections):
            if i not in used_det_indices:
                new_track_assignments[self.next_id] = det["bbox"]
                self.tracks[self.next_id] = {"bbox": det["bbox"], "age": 0}
                self.next_id += 1

        # 3. Clean up old tracks
        for tid in list(self.tracks.keys()):
            if tid in new_track_assignments:
                self.tracks[tid]["bbox"] = new_track_assignments[tid]
            elif self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        
        # Return only active tracks (age == 0) for display, but keep others in memory
        return {tid: self.tracks[tid] for tid in self.tracks if self.tracks[tid]["age"] == 0}

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
