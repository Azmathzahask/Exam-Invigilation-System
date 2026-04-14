from utils.logger import logger

class CrossCameraCorrelator:
    """
    Cross-camera correlation for dual-camera validation.
    Increases confidence when both cameras confirm suspicious behavior.
    """
    def __init__(self):
        self.confidence_boost = 1.5  # Multiplier for cross-camera confirmed events
        logger.info("Cross-camera correlator initialized with confidence boosting")

    def correlate(self, front_tracks, back_tracks, front_suspicious, back_suspicious):
        """
        Correlates observations between front and back cameras.
        
        Args:
            front_tracks: {id: {bbox, ...}} from front camera
            back_tracks: {id: {bbox, ...}} from back camera
            front_suspicious: [{id, type}, ...] from front camera analysis
            back_suspicious: [{id, type}, ...] from back camera analysis
        
        Returns:
            correlated_events: [{id, type, confidence}, ...]
        
        Logic:
        - Front camera: better for head direction (gaze detection)
        - Back camera: better for body leaning
        - Cross-confirm when both cameras detect same student ID with related behaviors
        """
        correlated_events = []
        
        # Create event maps by student ID
        front_events_by_id = {}
        for event in front_suspicious:
            tid = event["id"]
            if tid not in front_events_by_id:
                front_events_by_id[tid] = []
            front_events_by_id[tid].append(event["type"])
        
        back_events_by_id = {}
        for event in back_suspicious:
            tid = event["id"]
            if tid not in back_events_by_id:
                back_events_by_id[tid] = []
            back_events_by_id[tid].append(event["type"])
        
        # Find students detected by both cameras
        common_ids = set(front_events_by_id.keys()) & set(back_events_by_id.keys())
        
        for tid in common_ids:
            front_types = set(front_events_by_id[tid])
            back_types = set(back_events_by_id[tid])
            
            # Cross-camera confirmation scenarios:
            
            # 1. Side-looking (front) + Leaning (back) = High confidence copying
            if "side_look_frequent" in front_types and "leaning" in back_types:
                correlated_events.append({
                    "id": tid,
                    "type": "cross_camera_copying",
                    "confidence": "high",
                    "sources": ["front_gaze", "back_leaning"]
                })
                logger.info(f"Cross-camera confirmation: ID {tid} copying (gaze + leaning)")
            
            # 2. Talking detected by both cameras = High confidence interaction
            if "talking_proximity" in front_types and "talking_proximity" in back_types:
                correlated_events.append({
                    "id": tid,
                    "type": "cross_camera_talking",
                    "confidence": "high",
                    "sources": ["front_proximity", "back_proximity"]
                })
            
            # 3. Phone/prohibited object detected by both = Very high confidence
            prohibited_types = {"phone_detected", "chit_detected", "smartwatch_detected"}
            front_prohibited = front_types & prohibited_types
            back_prohibited = back_types & prohibited_types
            
            if front_prohibited and back_prohibited:
                for item_type in front_prohibited & back_prohibited:
                    correlated_events.append({
                        "id": tid,
                        "type": f"cross_camera_{item_type}",
                        "confidence": "very_high",
                        "sources": ["front_detection", "back_detection"]
                    })
        
        return correlated_events

    def boost_scores(self, scores, correlated_events):
        """
        Apply confidence boost to scores for cross-camera confirmed events.
        
        Args:
            scores: {id: score} from scoring system
            correlated_events: [{id, type, confidence}, ...] from correlate()
        
        Returns:
            boosted_scores: {id: score} with confidence multipliers applied
        """
        boosted_scores = scores.copy()
        
        for event in correlated_events:
            tid = event["id"]
            confidence = event["confidence"]
            
            if tid in boosted_scores:
                # Apply boost based on confidence level
                if confidence == "very_high":
                    boosted_scores[tid] *= 2.0
                elif confidence == "high":
                    boosted_scores[tid] *= self.confidence_boost
        
        return boosted_scores
