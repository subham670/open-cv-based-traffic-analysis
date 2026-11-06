"""
FlowPulse ITMS - Congestion Analysis & Level of Service (LOS) Engine
Implements Highway Capacity Manual (HCM) standards to compute traffic density,
flow stability, speed drops, and incident alerts.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
from backend.models.tracker import TrackedVehicle

class CongestionEngine:
    def __init__(self, speed_limit_kmh: float = 80.0, road_capacity_vehicles: int = 18):
        self.speed_limit_kmh = speed_limit_kmh
        self.road_capacity_vehicles = road_capacity_vehicles
        
        # Historical buffer for rolling metrics
        self.density_history = deque(maxlen=30)
        self.speed_history = deque(maxlen=30)
        self.congestion_history = deque(maxlen=30)
        
        # Incident thresholds
        self.stalled_threshold_sec = 4.5
        self.severe_congestion_threshold = 75.0

    def analyze(
        self,
        active_tracks: List[TrackedVehicle],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        h, w = frame_shape[:2]
        total_active = len(active_tracks)

        # 1. Road Density & Area Occupancy Ratio
        # Total vehicle bounding box area relative to total road area (approx bottom 75% of frame)
        road_area = w * h * 0.75
        vehicle_area = sum(t.box[2] * t.box[3] for t in active_tracks)
        occupancy_ratio = min(1.0, vehicle_area / max(1.0, road_area))

        # Density Ratio relative to optimal maximum capacity
        density_ratio = min(1.0, total_active / max(1, self.road_capacity_vehicles))

        # 2. Average Corridor Speed & Speed Deviation
        valid_speeds = [t.speed_kmh for t in active_tracks if t.speed_kmh > 0]
        if valid_speeds:
            avg_speed = float(np.mean(valid_speeds))
            min_speed = float(np.min(valid_speeds))
            max_speed = float(np.max(valid_speeds))
        else:
            avg_speed = self.speed_limit_kmh * 0.85
            min_speed = avg_speed
            max_speed = avg_speed

        # 3. Congestion Index (0% to 100%)
        # Weighted composite of:
        # - Capacity occupancy (45%)
        # - Speed reduction from free-flow speed limit (45%)
        # - Visual area occupancy (10%)
        speed_deficit_ratio = max(0.0, min(1.0, (self.speed_limit_kmh - avg_speed) / self.speed_limit_kmh))
        raw_congestion_index = (
            (density_ratio * 0.45) +
            (speed_deficit_ratio * 0.45) +
            (occupancy_ratio * 0.10)
        ) * 100.0

        # Smoothing with rolling average
        self.congestion_history.append(raw_congestion_index)
        congestion_index = round(float(np.mean(self.congestion_history)), 1)
        self.density_history.append(density_ratio)
        self.speed_history.append(avg_speed)

        # 4. Level of Service (LOS) Standard (A to F)
        los_level, los_description, los_color = self._determine_los(congestion_index, avg_speed)

        # 5. Incident & Anomaly Detection
        incidents = []
        for track in active_tracks:
            # Stalled Vehicle Alert
            if track.stationary_sec >= self.stalled_threshold_sec:
                incidents.append({
                    "type": "STALLED_VEHICLE",
                    "severity": "CRITICAL",
                    "track_id": track.track_id,
                    "lane_id": track.lane_id,
                    "duration_sec": round(track.stationary_sec, 1),
                    "message": f"Hazard: Vehicle #{track.track_id} stationary in Lane {track.lane_id} for {track.stationary_sec:.1f}s"
                })

            # Speeding Violation Alert
            if track.speed_kmh > (self.speed_limit_kmh + 10.0):
                incidents.append({
                    "type": "SPEED_VIOLATION",
                    "severity": "HIGH",
                    "track_id": track.track_id,
                    "lane_id": track.lane_id,
                    "speed_kmh": round(track.speed_kmh, 1),
                    "message": f"Speed Alert: Vehicle #{track.track_id} recorded at {track.speed_kmh:.1f} km/h (Limit: {self.speed_limit_kmh} km/h)"
                })

        # Bottleneck Alert
        if congestion_index >= self.severe_congestion_threshold:
            incidents.append({
                "type": "CONGESTION_BOTTLENECK",
                "severity": "HIGH",
                "track_id": None,
                "lane_id": None,
                "message": f"Traffic Bottleneck Detected: Congestion index reached {congestion_index}% ({los_description})"
            })

        return {
            "congestion_index": congestion_index,
            "density_ratio": round(density_ratio, 3),
            "occupancy_ratio": round(occupancy_ratio, 3),
            "avg_speed_kmh": round(avg_speed, 1),
            "min_speed_kmh": round(min_speed, 1),
            "max_speed_kmh": round(max_speed, 1),
            "los_level": los_level,
            "los_description": los_description,
            "los_color": los_color,
            "incidents": incidents
        }

    def _determine_los(self, congestion_index: float, avg_speed: float) -> Tuple[str, str, str]:
        """
        Highway Capacity Manual Level of Service Classification:
        LOS A: Free Flow, low density, high comfort.
        LOS B: Reasonably Free Flow, slight interaction.
        LOS C: Stable Flow, maneuverability restricted.
        LOS D: Approaching Unstable Flow, high density.
        LOS E: Unstable Flow / Capacity Limit, stop-and-go waves.
        LOS F: Forced Breakdown / Gridlock.
        """
        if congestion_index < 20:
            return "A", "Free Flow (Optimal)", "#10b981"  # Emerald
        elif congestion_index < 38:
            return "B", "Reasonably Free Flow", "#06b6d4"  # Cyan
        elif congestion_index < 55:
            return "C", "Stable Flow", "#f59e0b"          # Amber
        elif congestion_index < 72:
            return "D", "Approaching Unstable Flow", "#f97316" # Orange
        elif congestion_index < 88:
            return "E", "High Congestion / Capacity Limit", "#ef4444" # Red
        else:
            return "F", "Forced Breakdown / Gridlock", "#991b1b" # Dark Crimson
