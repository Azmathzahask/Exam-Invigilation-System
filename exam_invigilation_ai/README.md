# AI-Powered Dual-Camera Exam Invigilation System

A comprehensive AI system for monitoring exam halls using synchronized camera feeds.

## Features
- **Dual-Camera Sync**: Synchronized processing of front and back camera feeds.
- **YOLOv8 Detection**: Real-time detection of students, phones, and books.
- **Behavior Analysis**: Detects side-looking, leaning, and abnormal movements.
- **Progessive Scoring**: Suspicion points system with decay logic.
- **Live Dashboard**: Flask-based real-time monitoring with video streams and alerts.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your video files in `data/videos/front/` and `data/videos/back/`.
4. Update `config/config.yaml` with your file paths.

## Usage
Run the main pipeline:
```bash
python main.py
```
Open your browser at `http://localhost:5000` to view the dashboard.

## Folder Structure
- `modules/`: Core AI and analysis logic.
- `dashboard/`: Web interface components.
- `config/`: System thresholds and paths.
- `utils/`: Logging and helper functions.

## Privacy & Security
- No permanent storage of biometric data.
- Temporary IDs only per session.
- Only suspicious event clips are stored.
