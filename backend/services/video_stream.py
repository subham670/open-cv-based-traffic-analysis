"""
FlowPulse ITMS - Video Processing & Frame Annotation Engine
Orchestrates detector, tracker, congestion engine, and telemetry HUD overlays.
"""
import cv2
import numpy as np
import time
import os
from typing import Dict, Any, Generator, Optional
import logging

from backend.models.detector import VehicleDetector
from backend.models.tracker import VehicleTracker
from backend.models.congestion import CongestionEngine
from backend.models.forecaster import TrafficForecaster
from backend.services.generator import TrafficSceneGenerator

logger = logging.getLogger("flowpulse.video_stream")

class VideoStreamProcessor:
    def __init__(self):
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker()
        self.congestion_engine = CongestionEngine()
        self.forecaster = TrafficForecaster()
        
        self.current_source_type = "synthetic_highway"
        self.custom_video_path = ""
        self.cap = None
        self.synthetic_gen = TrafficSceneGenerator(width=960, height=540, mode="highway")
        
        # Telemetry State Cache
        self.latest_telemetry: Dict[str, Any] = {
            "fps": 30.0,
            "active_vehicles": 0,
            "total_in": 0,
            "total_out": 0,
            "avg_speed_kmh": 0.0,
            "congestion_index": 0.0,
            "los_level": "A",
            "los_description": "Free Flow",
            "los_color": "#10b981",
            "class_counts": {},
            "lane_counts": {},
            "incidents": [],
            "source_type": self.current_source_type
        }

        # Overlay display toggles
        self.show_boxes = True
        self.show_trails = True
        self.show_tripwires = True
        self.show_speed = True
        self.show_hud = True
        self.show_lanes = True

        self.last_frame_time = time.time()
        self.fps_smoothed = 30.0

    def switch_source(self, source_type: str, file_path: str = "", webcam_idx: int = 0):
        logger.info(f"Switching video source to: {source_type} ({file_path})")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.current_source_type = source_type
        self.custom_video_path = file_path

        if source_type == "webcam":
            self.cap = cv2.VideoCapture(webcam_idx)
        elif source_type == "custom_file" and os.path.exists(file_path):
            self.cap = cv2.VideoCapture(file_path)
        elif source_type == "synthetic_intersection":
            self.synthetic_gen = TrafficSceneGenerator(width=960, height=540, mode="intersection")
        else:
            self.synthetic_gen = TrafficSceneGenerator(width=960, height=540, mode="highway")

        # Reset counts on source switch
        self.tracker.reset_counts()

    def set_display_options(self, opts: Dict[str, bool]):
        if "show_boxes" in opts:
            self.show_boxes = opts["show_boxes"]
        if "show_trails" in opts:
            self.show_trails = opts["show_trails"]
        if "show_tripwires" in opts:
            self.show_tripwires = opts["show_tripwires"]
        if "show_speed" in opts:
            self.show_speed = opts["show_speed"]
        if "show_hud" in opts:
            self.show_hud = opts["show_hud"]
        if "show_lanes" in opts:
            self.show_lanes = opts["show_lanes"]

    def set_tripwires(self, line_in_y: int, line_out_y: int):
        self.tracker.set_tripwire_lines(line_in_y, line_out_y)

    def process_next_frame(self) -> np.ndarray:
        current_time = time.time()
        dt = max(0.001, current_time - self.last_frame_time)
        self.last_frame_time = current_time
        instant_fps = 1.0 / dt
        self.fps_smoothed = self.fps_smoothed * 0.9 + instant_fps * 0.1

        # 1. Fetch raw frame from active source
        raw_frame = None
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                raw_frame = cv2.resize(frame, (960, 540))
            else:
                # Loop video file if ended
                if self.custom_video_path and os.path.exists(self.custom_video_path):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret2, frame2 = self.cap.read()
                    if ret2:
                        raw_frame = cv2.resize(frame2, (960, 540))

        if raw_frame is None:
            raw_frame = self.synthetic_gen.generate_frame()

        h, w = raw_frame.shape[:2]

        # 2. Vehicle Detection
        detections = self.detector.detect(raw_frame)

        # 3. Vehicle Tracking & Tripwire Counting
        tracked_vehicles = self.tracker.update(detections, (h, w), current_time)

        # 4. Congestion & Level of Service Analysis
        congestion_data = self.congestion_engine.analyze(tracked_vehicles, (h, w))

        # 5. Record observation to ML forecaster
        self.forecaster.record_observation(
            count_delta=self.tracker.total_in + self.tracker.total_out,
            current_speed=congestion_data["avg_speed_kmh"]
        )

        # Update Telemetry Snapshot Cache
        self.latest_telemetry = {
            "fps": round(self.fps_smoothed, 1),
            "active_vehicles": len(tracked_vehicles),
            "total_in": self.tracker.total_in,
            "total_out": self.tracker.total_out,
            "avg_speed_kmh": congestion_data["avg_speed_kmh"],
            "min_speed_kmh": congestion_data["min_speed_kmh"],
            "max_speed_kmh": congestion_data["max_speed_kmh"],
            "congestion_index": congestion_data["congestion_index"],
            "los_level": congestion_data["los_level"],
            "los_description": congestion_data["los_description"],
            "los_color": congestion_data["los_color"],
            "class_counts": dict(self.tracker.class_counts),
            "lane_counts": dict(self.tracker.lane_counts),
            "incidents": congestion_data["incidents"],
            "source_type": self.current_source_type
        }

        # 6. Render Overlays on Frame
        annotated_frame = raw_frame.copy()

        # Overlay Lane Markers
        if self.show_lanes:
            self._render_lane_zones(annotated_frame, w, h)

        # Overlay Tripwires
        if self.show_tripwires:
            self._render_tripwires(annotated_frame, w, h)

        # Overlay Vehicle Tracks, Motion Trails & Bounding Boxes
        for track in tracked_vehicles:
            if self.show_trails:
                self._render_trails(annotated_frame, track)
            if self.show_boxes:
                self._render_vehicle_box(annotated_frame, track)

        # Overlay Top Enterprise Telemetry HUD
        if self.show_hud:
            self._render_hud(annotated_frame, w, h)

        return annotated_frame

    def _render_lane_zones(self, frame: np.ndarray, w: int, h: int):
        # Subtle lane boundaries
        for l in range(1, 4):
            lx = int(l * (w / 4))
            cv2.line(frame, (lx, int(h * 0.15)), (lx, h), (70, 70, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, f"LANE {l}", (lx - 75, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, "LANE 4", (w - 75, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 160), 1, cv2.LINE_AA)

    def _render_tripwires(self, frame: np.ndarray, w: int, h: int):
        # Line A: Inbound (Cyan Glow)
        y_in = self.tracker.line_in_y
        cv2.line(frame, (20, y_in), (w - 20, y_in), (255, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (35, y_in), 5, (255, 220, 0), -1)
        cv2.putText(frame, f"TRIPWIRE A (INBOUND): {self.tracker.total_in}", (50, y_in - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 50), 1, cv2.LINE_AA)

        # Line B: Outbound (Amber Glow)
        y_out = self.tracker.line_out_y
        cv2.line(frame, (20, y_out), (w - 20, y_out), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (35, y_out), 5, (0, 165, 255), -1)
        cv2.putText(frame, f"TRIPWIRE B (OUTBOUND): {self.tracker.total_out}", (50, y_out - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 195, 255), 1, cv2.LINE_AA)

    def _render_trails(self, frame: np.ndarray, track):
        history = list(track.history)
        if len(history) < 2:
            return
        
        trail_color = (0, 240, 255) if track.direction == "SOUTHBOUND_IN" else (255, 120, 0)
        for i in range(1, len(history)):
            pt1 = (history[i-1][0], history[i-1][1])
            pt2 = (history[i][0], history[i][1])
            thickness = max(1, int(i / len(history) * 3))
            cv2.line(frame, pt1, pt2, trail_color, thickness, cv2.LINE_AA)

    def _render_vehicle_box(self, frame: np.ndarray, track):
        x, y, bw, bh = track.box
        cx, cy = track.centroid

        # Color coding by speed & class
        if track.stationary_sec > 4.0:
            box_color = (0, 0, 255) # Red for hazard/stalled
        elif track.speed_kmh > 80.0:
            box_color = (0, 140, 255) # Amber/Orange for speeding
        else:
            box_color = (50, 220, 100) # Green for regular flow

        # Sleek Corner Reticle styling
        corner_len = min(14, int(min(bw, bh) * 0.35))
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), box_color, 2)
        cv2.line(frame, (x, y), (x, y + corner_len), box_color, 2)
        # Top-right
        cv2.line(frame, (x + bw, y), (x + bw - corner_len, y), box_color, 2)
        cv2.line(frame, (x + bw, y), (x + bw, y + corner_len), box_color, 2)
        # Bottom-left
        cv2.line(frame, (x, y + bh), (x + corner_len, y + bh), box_color, 2)
        cv2.line(frame, (x, y + bh), (x, y + bh - corner_len), box_color, 2)
        # Bottom-right
        cv2.line(frame, (x + bw, y + bh), (x + bw - corner_len, y + bh), box_color, 2)
        cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - corner_len), box_color, 2)
        
        # Semi-transparent box border
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 1)

        # Vehicle Label Tag: ID, Class, Speed
        label = f"#{track.track_id} {track.class_name.upper()}"
        if self.show_speed and track.speed_kmh > 0:
            label += f" | {track.speed_kmh:.0f} km/h"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        # Tag background banner
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), (20, 20, 25), -1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), box_color, 1)
        cv2.putText(frame, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    def _render_hud(self, frame: np.ndarray, w: int, h: int):
        # Top Modern Telemetry HUD Banner
        hud_h = 42
        hud_bg = frame[0:hud_h, 0:w].copy()
        cv2.rectangle(frame, (0, 0), (w, hud_h), (16, 20, 26), -1)
        cv2.addWeighted(hud_bg, 0.25, frame[0:hud_h, 0:w], 0.75, 0, frame[0:hud_h, 0:w])
        cv2.line(frame, (0, hud_h), (w, hud_h), (40, 50, 70), 1)

        # Items in HUD: Status dot, Source, Density/LOS, FPS, Time
        cv2.circle(frame, (20, 21), 6, (50, 220, 80), -1)
        cv2.putText(frame, "LIVE REC", (34, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)
        
        # Active Vehicles
        active_str = f"ACTIVE TARGETS: {self.latest_telemetry['active_vehicles']}"
        cv2.putText(frame, active_str, (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        # LOS Badge
        los_str = f"LOS: {self.latest_telemetry['los_level']} ({self.latest_telemetry['congestion_index']}%)"
        cv2.putText(frame, los_str, (370, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 240, 150), 1, cv2.LINE_AA)

        # Avg Speed
        speed_str = f"AVG SPD: {self.latest_telemetry['avg_speed_kmh']:.1f} KM/H"
        cv2.putText(frame, speed_str, (600, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 200, 50), 1, cv2.LINE_AA)

        # FPS
        fps_str = f"FPS: {self.latest_telemetry['fps']:.1f}"
        cv2.putText(frame, fps_str, (850, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 200), 1, cv2.LINE_AA)

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        while True:
            frame = self.process_next_frame()
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.02)  # Maintain smooth ~30-40 FPS
