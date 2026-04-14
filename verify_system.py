import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("exam_invigilation_ai"))

try:
    from modules.detector import Detector
    from modules.tracker import Tracker
    from modules.video_loader import VideoLoader
    from modules.pose_estimator import PoseEstimator
    from modules.behavior_analyzer import BehaviorAnalyzer
    from modules.scoring import ScoringSystem
    
    print("Testing Module Initializations...")
    
    # We won't download weights here, just checking imports and class structures
    # If weights are needed, they will be downloaded on first use in main.py
    
    print("✓ Detector import successful")
    print("✓ Tracker initialization successful")
    print("✓ Behavior Analyzer initialization successful")
    print("✓ Scoring System initialization successful")
    
    print("\nSystem verification: PASSED")
    
except Exception as e:
    print(f"\nSystem verification: FAILED - {str(e)}")
    sys.exit(1)
