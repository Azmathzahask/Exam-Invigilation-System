import cv2
import numpy as np

def resize_frame(frame, width=640, height=480):
    """Resizes frame to set dimensions."""
    return cv2.resize(frame, (width, height))

def get_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(a, b, c):
    """Calculates angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle
