import pytest
import sys
import os
import yaml

# Add project root to path
sys.path.append(os.path.abspath("exam_invigilation_ai"))

from modules.scoring import ScoringSystem
from modules.behavior_analyzer import BehaviorAnalyzer

def test_scoring_system():
    scorer = ScoringSystem(weights={"phone_detected": 20}, decay_rate=0.1)
    # Add points
    scores = scorer.update_scores([{"id": 1, "type": "phone_detected"}], {1: {"bbox": [0,0,10,10]}})
    assert scores[1] == 20
    
    # Check decay
    scores = scorer.update_scores([], {1: {"bbox": [0,0,10,10]}})
    assert scores[1] == pytest.approx(19.9, abs=0.5)

def test_behavior_analyzer_init():
    analyzer = BehaviorAnalyzer(side_look_threshold=30)
    assert analyzer.side_look_threshold == 30

def test_config_loading():
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    assert 'video' in config
    assert 'detection' in config
