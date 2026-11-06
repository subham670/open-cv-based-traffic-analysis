"""
FlowPulse ITMS - System Configuration
"""
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    APP_NAME: str = "FlowPulse Intelligent Traffic Management System (ITMS)"
    APP_VERSION: str = "2.4.0-PROD"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Detection Engine Settings
    DETECTOR_MODEL_TYPE: str = "yolov4_tiny"  # 'yolov4_tiny', 'onnx', or 'hybrid_cv'
    MODEL_WEIGHTS_PATH: str = os.path.join("models", "yolov4-tiny.weights")
    MODEL_CFG_PATH: str = os.path.join("models", "yolov4-tiny.cfg")
    MODEL_NAMES_PATH: str = os.path.join("models", "coco.names")
    
    CONFIDENCE_THRESHOLD: float = 0.45
    NMS_THRESHOLD: float = 0.40
    INPUT_SIZE: tuple = (416, 416)

    # Vehicle COCO Classes to Track
    TARGET_VEHICLE_CLASSES: List[str] = [
        "car", "motorbike", "bus", "truck", "bicycle"
    ]

    # Tracking & Speed Calibration
    MAX_DISAPPEARED_FRAMES: int = 25
    MAX_TRACK_DISTANCE: float = 85.0
    FPS_DEFAULT: float = 30.0
    PIXELS_TO_METERS_RATIO: float = 0.08  # Calibrated for typical highway camera perspective
    SPEED_LIMIT_KMH: float = 80.0         # Speed limit threshold for violations

    # Congestion & Level of Service (HCM Standards)
    # LOS Density thresholds (vehicles per km or road occupancy ratio)
    LOS_THRESHOLDS: Dict[str, float] = {
        "A": 0.15,  # Free flow
        "B": 0.30,  # Reasonably free flow
        "C": 0.50,  # Stable flow
        "D": 0.70,  # Approaching unstable
        "E": 0.85,  # Unstable / High congestion
        "F": 1.00   # Gridlock / Breakdown
    }

    # Video Streams Catalog
    AVAILABLE_CAMERAS: List[Dict[str, Any]] = [
        {
            "id": "cam_highway",
            "name": "CAM-01: Metropolitan Arterial Highway (North)",
            "type": "synthetic_highway",
            "speed_limit": 80,
            "lanes": 4,
            "direction": "Dual-Carriageway"
        },
        {
            "id": "cam_intersection",
            "name": "CAM-02: Downtown Central Cross-Junction",
            "type": "synthetic_intersection",
            "speed_limit": 50,
            "lanes": 4,
            "direction": "Multi-Directional"
        },
        {
            "id": "cam_expressway",
            "name": "CAM-03: Outer Ring Expressway (Fast Lane)",
            "type": "synthetic_expressway",
            "speed_limit": 100,
            "lanes": 3,
            "direction": "Inbound Expressway"
        },
        {
            "id": "cam_webcam",
            "name": "CAM-04: Live Optical Sensor / USB Webcam",
            "type": "webcam",
            "device_index": 0
        },
        {
            "id": "cam_custom",
            "name": "CAM-05: User Uploaded Video Source",
            "type": "custom_file",
            "file_path": ""
        }
    ]

    DATABASE_URL: str = "sqlite:///backend/data/traffic_analytics.db"

settings = Settings()
