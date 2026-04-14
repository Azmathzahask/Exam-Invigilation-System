import yaml
import time
import threading
import cv2
import numpy as np
from utils.logger import logger
from modules.video_loader import VideoLoader
from modules.detector import Detector
from modules.tracker import Tracker
from modules.pose_estimator import PoseEstimator
from modules.behavior_analyzer import BehaviorAnalyzer
from modules.scoring import ScoringSystem
from modules.cross_camera import CrossCameraCorrelator
from dashboard.app import run_flask, frame_buffer, event_log, student_scores

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# The system dynamically maps back camera to front camera IDs frame-by-frame


def extract_histogram(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def main():
    config = load_config()
    
    # Initialize Core modules
    video_loader = VideoLoader(
        config['video']['front_path'],
        config['video']['back_path'],
        (config['video']['frame_width'], config['video']['frame_height'])
    )
    
    detector = Detector(
        config['detection']['model_path'],
        config['detection']['confidence_threshold'],
        config['detection']['target_classes']
    )
    
    tracker_f = Tracker(config['tracking']['algorithm'])
    tracker_b = Tracker(config['tracking']['algorithm'])
    pose_estimator = PoseEstimator()
    behavior_analyzer = BehaviorAnalyzer(
        config['behavior']['side_look_threshold'],
        config['behavior']['side_look_count'],
        config['behavior']['time_window']
    )
    
    scoring_system = ScoringSystem(
        config['scoring']['weights'],
        config['scoring']['decay_rate'],
        config['scoring']['suspicion_threshold'],
        config['scoring']['alert_threshold']
    )
    
    correlator = CrossCameraCorrelator()

    # Start dashboard in separate thread
    flask_thread = threading.Thread(target=run_flask, args=(config['dashboard']['port'],))
    flask_thread.daemon = True
    flask_thread.start()
    logger.info(f"Dashboard started on port {config['dashboard']['port']}")

    # Start video loading
    video_loader.start()

    try:
        logger.info("Starting main processing loop")
        while True:
            try:
                # Catch up to the latest frame if the queue is building up
                # This ensures the dashboard stays "Live" and doesn't lag/freeze
                qsize = video_loader.front_queue.qsize()
                if qsize > 1:
                    for _ in range(qsize - 1):
                        video_loader.front_queue.get_nowait()
                
                qsize_b = video_loader.back_queue.qsize()
                if qsize_b > 1:
                    for _ in range(qsize_b - 1):
                        video_loader.back_queue.get_nowait()

                frame_f, frame_b = video_loader.get_synced_frames()
                
                if frame_f is None:
                    time.sleep(0.01)
                    continue

                # Process Front Camera
                detections_f = detector.detect(frame_f)
                tracks_f = tracker_f.update(detections_f, frame_f)
                gaze_f = pose_estimator.estimate_gaze(frame_f, tracks_f)
                
                # Process Back Camera
                if frame_b is not None and not np.array_equal(frame_b, np.zeros_like(frame_b)):
                    detections_b = detector.detect(frame_b)
                    tracks_b = tracker_b.update(detections_b, frame_b)
                else:
                    tracks_b = {}
                
                # Update scores
                suspicious_events = behavior_analyzer.analyze(tracks_f, gaze_f, detections_f)
                current_scores = scoring_system.update_scores(suspicious_events, tracks_f)

                # Draw detections on frame_f for dashboard
                for tid, tdata in tracks_f.items():
                    bbox = tdata["bbox"]
                    status = scoring_system.get_status(tid)
                    score = current_scores.get(tid, 0)
                    
                    # Color logic: RED for ALERT, Orange for WATCHLIST, Green for NORMAL
                    if status == "ALERT":
                        color = (0, 0, 255) # Red in BGR
                    elif status == "WATCHLIST":
                        color = (0, 165, 255) # Orange in BGR
                    else:
                        color = (0, 255, 0) # Green in BGR
                    
                    cv2.rectangle(frame_f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.putText(frame_f, f"ID:{tid}", (int(bbox[0]), int(bbox[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update Dashboard shared data
                if frame_f is not None:
                    frame_buffer["front"] = frame_f.copy()
                    if int(time.time()) % 10 == 0: 
                        logger.info(f"Frame buffer 'front' updated. Queue size: {video_loader.front_queue.qsize()}")
                    
                if frame_b is not None:
                    # Dynamically auto-map back-IDs to front-IDs via Fused Spatio-Chromal Cost Matrix EVERY FRAME
                    current_mapping = {}
                    
                    if tracks_b and tracks_f:
                        h_b, w_b = frame_b.shape[:2]
                        h_f, w_f = frame_f.shape[:2]
                        
                        matches = []
                        for tid_b, tdata_b in tracks_b.items():
                            hist_b = extract_histogram(frame_b, tdata_b["bbox"])
                            if hist_b is None: continue
                            
                            for tid_f, tdata_f in tracks_f.items():
                                hist_f = extract_histogram(frame_f, tdata_f["bbox"])
                                if hist_f is None: continue
                                
                                # Spatial distance (Y is inverted, X is same)
                                bbox_b = tdata_b["bbox"]
                                cx_b = (bbox_b[0] + bbox_b[2]) / 2 / w_b
                                cy_b = (bbox_b[1] + bbox_b[3]) / 2 / h_b
                                
                                bbox_f = tdata_f["bbox"]
                                cx_f = (bbox_f[0] + bbox_f[2]) / 2 / w_f
                                cy_f = (bbox_f[1] + bbox_f[3]) / 2 / h_f
                                
                                spatial_dist = ((cx_f - cx_b)**2 + (cy_f - (1.0 - cy_b))**2)**0.5
                                
                                match_val = cv2.compareHist(hist_b, hist_f, cv2.HISTCMP_CORREL)
                                color_dist = 1.0 - match_val
                                
                                # Cost is a mix of geometrical similarity and clothing color similarity
                                total_cost = spatial_dist + (color_dist * 0.5)
                                matches.append((total_cost, tid_b, tid_f))
                                
                        # Greedy matching: lowest cost first
                        matches.sort(key=lambda x: x[0])
                        used_f = set()
                        
                        for total_cost, tid_b, tid_f in matches:
                            if tid_b not in current_mapping and tid_f not in used_f:
                                if total_cost < 0.8: # Tolerance map
                                    current_mapping[tid_b] = tid_f
                                    used_f.add(tid_f)

                    # Draw detections on back camera with dynamically mapped IDs
                    for tid_b, tdata_b in tracks_b.items():
                        bbox = tdata_b["bbox"]
                        
                        mapped_tid = current_mapping.get(tid_b)
                        
                        if mapped_tid is not None:
                            status = scoring_system.get_status(mapped_tid)
                            if status == "ALERT":
                                color = (0, 0, 255) # Red
                            elif status == "WATCHLIST":
                                color = (0, 165, 255) # Orange
                            else:
                                color = (0, 255, 0) # Green
                            display_text = f"ID:{mapped_tid}"
                        else:
                            color = (0, 255, 0) # Default to green normal if unmapped, no cyan boxes
                            display_text = f"ID:B{tid_b}"
                            
                        cv2.rectangle(frame_b, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.putText(frame_b, display_text, (int(bbox[0]), int(bbox[1]-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                    frame_buffer["back"] = frame_b.copy()
                else:
                    # If back is missing, we use a placeholder or empty frame if first run
                    if frame_buffer.get("back") is None and frame_f is not None:
                        frame_buffer["back"] = np.zeros_like(frame_f)
                
                if suspicious_events:
                    logger.info(f"Events: {suspicious_events}")
                    for event in suspicious_events:
                        event_msg = {"timestamp": time.time(), "id": event["id"], "type": event["type"], "description": event.get("description", "")}
                        event_log.append(event_msg)
                
                student_scores.update(current_scores)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(1) # Wait a bit before retrying

    except KeyboardInterrupt:
        logger.info("System shutting down...")
    finally:
        video_loader.stop()

if __name__ == "__main__":
    main()
