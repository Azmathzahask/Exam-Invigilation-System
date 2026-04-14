import logging
from flask import Flask, render_template, Response, jsonify
import cv2
import queue
import threading
import time
from utils.logger import logger

app = Flask(__name__)

# Shared data structures for dashboard updates
frame_buffer = {"front": None, "back": None}
event_log = []
student_scores = {}

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(camera_type):
    logger.info(f"Starting stream for {camera_type}")
    while True:
        frame = frame_buffer.get(camera_type)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error(f"Failed to encode frame for {camera_type}")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed/<camera_type>')
def video_feed(camera_type):
    return Response(gen_frames(camera_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify({
        "events": event_log[-10:], # Return last 10 events
        "scores": student_scores,
        "timestamp": time.time()
    })

def run_flask(port=5000):
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_flask()
