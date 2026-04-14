import cv2
import threading
import queue
import time
from utils.logger import logger

class VideoLoader:
    def __init__(self, front_path, back_path, frame_size=(640, 480), sync_tolerance=0.03):
        self.front_path = front_path
        self.back_path = back_path
        self.frame_size = frame_size
        self.sync_tolerance = sync_tolerance
        
        self.front_cap = cv2.VideoCapture(front_path)
        self.back_cap = cv2.VideoCapture(back_path)
        
        self.front_queue = queue.Queue(maxsize=30)
        self.back_queue = queue.Queue(maxsize=30)
        
        self.running = False
        self.front_thread = None
        self.back_thread = None

    def _read_frames(self, cap, q, name):
        logger.info(f"Starting frame reader for {name}")
        if not cap.isOpened():
            logger.error(f"Failed to open video source for {name}: {self.front_path if name == 'Front' else self.back_path}")
            return

        while self.running:
            if not q.full():
                ret, frame = cap.read()
                if not ret:
                    # Loop video: Reset to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        logger.error(f"Failed to restart video for {name}")
                        break
                    logger.info(f"Restarting video for {name} (looping active)")
                
                # Log ogni 100 frame
                frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if frame_count % 100 == 0:
                    logger.debug(f"{name} camera processed {int(frame_count)} frames")

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame = cv2.resize(frame, self.frame_size)
                q.put((timestamp, frame))
            else:
                time.sleep(0.01)

    def start(self):
        self.running = True
        self.front_thread = threading.Thread(target=self._read_frames, args=(self.front_cap, self.front_queue, "Front"))
        self.back_thread = threading.Thread(target=self._read_frames, args=(self.back_cap, self.back_queue, "Back"))
        self.front_thread.start()
        self.back_thread.start()
        logger.info("Video loader threads started")

    def get_synced_frames(self):
        try:
            ts_f, frame_f = self.front_queue.get(timeout=0.1)
        except queue.Empty:
            return None, None
        
        # Try to find a matching frame in back queue
        frame_b = None
        try:
            ts_b, frame_b = self.back_queue.get(timeout=0.1)
        except queue.Empty:
            pass
            
        return frame_f, frame_b

    def stop(self):
        self.running = False
        if self.front_thread: self.front_thread.join()
        if self.back_thread: self.back_thread.join()
        self.front_cap.release()
        self.back_cap.release()
        logger.info("Video loader stopped")
