from utils.logger import logger

class ScoringSystem:
    def __init__(self, weights=None, decay_rate=0.3, suspicion_threshold=50, alert_threshold=100):
        self.scores = {}  # {id: score}
        self.suspicion_threshold = suspicion_threshold
        self.alert_threshold = alert_threshold
        self.weights = weights or {
            "side_look": 5,
            "leaning": 10,
            "phone": 50,
            "smartwatch": 40,
            "chit": 45,
            "talking": 15,
            "look_back": 8,
            "abnormal_movement": 15
        }
        self.decay_rate = decay_rate  # Points per second
        self.last_update_time = {}  # {id: timestamp}
        logger.info("Scoring system initialized")

    def update_scores(self, suspicious_events, tracks):
        import time
        current_time = time.time()
        
        # 1. Apply time-based decay
        for tid in list(self.scores.keys()):
            if tid in self.last_update_time:
                time_elapsed = current_time - self.last_update_time[tid]
                decay_amount = self.decay_rate * time_elapsed
                self.scores[tid] = max(0, self.scores[tid] - decay_amount)
            self.last_update_time[tid] = current_time
            
        # 2. Add points for new suspicious events
        for event in suspicious_events:
            tid = event["id"]
            etype = event["type"]
            if tid not in self.scores:
                self.scores[tid] = 0
                self.last_update_time[tid] = current_time
            self.scores[tid] += self.weights.get(etype, 0)
            
            # Log score updates
            if self.scores[tid] > 0:
                logger.debug(f"ID {tid} score: {self.scores[tid]:.1f} (+{self.weights.get(etype, 0)} for {etype})")
            
        # 3. Remove scores for students no longer tracked
        current_ids = set(tracks.keys())
        self.scores = {tid: score for tid, score in self.scores.items() if tid in current_ids}
        self.last_update_time = {tid: t for tid, t in self.last_update_time.items() if tid in current_ids}
        
        return self.scores

    def get_status(self, tid):
        """
        Returns status:
        - ALERT: 50+ points (RED box) - 5 side-looks OR 1 phone + some side-looks
        - WATCHLIST: 25+ points (ORANGE box) - 2-3 side-looks
        - NORMAL: <25 points (GREEN box)
        """
        score = self.scores.get(tid, 0)
        if score >= self.alert_threshold:
            return "ALERT"
        if score >= self.suspicion_threshold:
            return "WATCHLIST"
        return "NORMAL"
