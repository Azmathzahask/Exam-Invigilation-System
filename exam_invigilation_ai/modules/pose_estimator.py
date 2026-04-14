import cv2
import numpy as np
from utils.logger import logger

class PoseEstimator:
    """
    Estimates 'gaze' or side-leaning by tracking the student's center-of-mass deviation
    from their seated baseline. This ignores neighboring students entirely to prevent false-flags.
    """
    def __init__(self):
        self.baselines = {} # {tid: {'x': int, 'y': int}}
        logger.info("PoseEstimator initialized - deviation-aware detection")

    def estimate_gaze(self, frame, tracks):
        gaze_data = {}
        
        for tid, tdata in tracks.items():
            bbox = tdata["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if tid not in self.baselines:
                self.baselines[tid] = {'x': center_x, 'y': center_y}
                gaze_data[tid] = "center"
                continue
                
            # Update baseline slowly (exponential moving average) so gradual seating changes are ignored
            bl = self.baselines[tid]
            bl['x'] = 0.98 * bl['x'] + 0.02 * center_x
            bl['y'] = 0.98 * bl['y'] + 0.02 * center_y
            
            # Check horizontal deviation
            dx = center_x - bl['x']
            
            # If they shifted horizontally by more than 80 pixels from their personal baseline
            if dx > 80:
                gaze_data[tid] = "right"
            elif dx < -80:
                gaze_data[tid] = "left"
            else:
                gaze_data[tid] = "center"
                
        return gaze_data
