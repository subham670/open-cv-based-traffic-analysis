"""
FlowPulse ITMS - High Precision Multi-Object Vehicle Tracker
Maintains vehicle trajectories, tripwire crossing counters, lane assignments, and dwell times.
"""
import numpy as np
from collections import OrderedDict, deque
from typing import List, Dict, Any, Tuple, Optional
import time
from backend.models.detector import VehicleDetection

class TrackedVehicle:
    def __init__(self, track_id: int, detection: VehicleDetection, current_time: float):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.box = detection.box  # [x, y, w, h]
        self.confidence = detection.confidence
        self.centroid = detection.center
        self.history = deque(maxlen=30)  # Stores (x, y, timestamp)
        self.history.append((self.centroid[0], self.centroid[1], current_time))
        
        self.first_seen = current_time
        self.last_seen = current_time
        self.disappeared = 0
        
        # Velocity & Motion
        self.speed_kmh = 0.0
        self.speed_history = deque(maxlen=10)
        self.stationary_duration = 0.0
        self.direction = "UNKNOWN"
        self.lane_id = 1
        
        # Line Crossing flags
        self.counted_in = False
        self.counted_out = False
        self.crossing_timestamp = None

    @property
    def stationary_sec(self) -> float:
        return self.stationary_duration

    def update(self, detection: VehicleDetection, current_time: float, pixels_to_meters: float = 0.08):
        dt = max(0.001, current_time - self.last_seen)
        old_cx, old_cy = self.centroid
        new_cx, new_cy = detection.center

        # Estimate instantaneous velocity
        pixel_distance = np.sqrt((new_cx - old_cx)**2 + (new_cy - old_cy)**2)
        meter_distance = pixel_distance * pixels_to_meters
        instant_speed_ms = meter_distance / dt
        instant_speed_kmh = instant_speed_ms * 3.6

        if pixel_distance < 3.0:
            self.stationary_duration += dt
        else:
            self.stationary_duration = 0.0

        # Smooth speed filter
        if 5.0 < instant_speed_kmh < 180.0:
            self.speed_history.append(instant_speed_kmh)
            self.speed_kmh = float(np.median(self.speed_history))
        elif len(self.speed_history) > 0:
            self.speed_kmh = float(np.mean(self.speed_history))

        # Direction detection based on Y-axis displacement
        if len(self.history) >= 3:
            dy = new_cy - self.history[0][1]
            if dy > 15:
                self.direction = "SOUTHBOUND_IN"
            elif dy < -15:
                self.direction = "NORTHBOUND_OUT"

        self.box = detection.box
        self.centroid = (new_cx, new_cy)
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.last_seen = current_time
        self.disappeared = 0
        self.history.append((new_cx, new_cy, current_time))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "box": self.box,
            "centroid": self.centroid,
            "speed_kmh": round(self.speed_kmh, 1),
            "lane_id": self.lane_id,
            "direction": self.direction,
            "stationary_sec": round(self.stationary_duration, 1),
            "trail": [(p[0], p[1]) for p in list(self.history)[-10:]]
        }

class VehicleTracker:
    def __init__(
        self,
        max_disappeared: int = 25,
        max_distance: float = 85.0,
        pixels_to_meters: float = 0.08
    ):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedVehicle] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.pixels_to_meters = pixels_to_meters

        # Tripwire Counting Lines (normalized y coords or absolute y coords)
        # Default: Line A (Inbound: y=380), Line B (Outbound: y=260)
        self.line_in_y = 380
        self.line_out_y = 260
        self.total_in = 0
        self.total_out = 0
        self.class_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "bicycle": 0}
        self.lane_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    def set_tripwire_lines(self, line_in_y: int, line_out_y: int):
        self.line_in_y = line_in_y
        self.line_out_y = line_out_y

    def update(
        self,
        detections: List[VehicleDetection],
        frame_shape: Tuple[int, int],
        current_time: Optional[float] = None
    ) -> List[TrackedVehicle]:
        if current_time is None:
            current_time = time.time()
            
        h, w = frame_shape[:2]

        # Dynamic tripwire placement if not explicitly set
        if self.line_in_y > h:
            self.line_in_y = int(h * 0.65)
        if self.line_out_y > h:
            self.line_out_y = int(h * 0.40)

        # Handle empty detections
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    del self.tracks[track_id]
            return list(self.tracks.values())

        # If no existing tracks, register all detections as new
        if len(self.tracks) == 0:
            for det in detections:
                self._register_track(det, current_time, w)
            return list(self.tracks.values())

        # Match detections to existing tracks via Euclidean Distance & IoU cost
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        detection_centroids = [det.center for det in detections]

        D = np.linalg.norm(np.array(track_centroids)[:, np.newaxis] - np.array(detection_centroids), axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            track_id = track_ids[row]
            det = detections[col]
            track = self.tracks[track_id]
            
            # Determine Lane ID based on X coordinate
            lane_width = w / 4.0
            track.lane_id = min(4, max(1, int(det.center[0] // lane_width) + 1))
            
            # Check Tripwire Crossing
            prev_y = track.centroid[1]
            curr_y = det.center[1]

            # Inbound crossing (Moving Down: prev_y < line_in_y <= curr_y)
            if not track.counted_in and prev_y < self.line_in_y <= curr_y:
                track.counted_in = True
                track.crossing_timestamp = current_time
                self.total_in += 1
                self.class_counts[track.class_name] = self.class_counts.get(track.class_name, 0) + 1
                self.lane_counts[track.lane_id] = self.lane_counts.get(track.lane_id, 0) + 1

            # Outbound crossing (Moving Up: prev_y > self.line_out_y >= curr_y)
            elif not track.counted_out and prev_y > self.line_out_y >= curr_y:
                track.counted_out = True
                track.crossing_timestamp = current_time
                self.total_out += 1
                self.class_counts[track.class_name] = self.class_counts.get(track.class_name, 0) + 1
                self.lane_counts[track.lane_id] = self.lane_counts.get(track.lane_id, 0) + 1

            track.update(det, current_time, self.pixels_to_meters)
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        # Mark disappeared tracks
        for row in unused_rows:
            track_id = track_ids[row]
            self.tracks[track_id].disappeared += 1
            if self.tracks[track_id].disappeared > self.max_disappeared:
                del self.tracks[track_id]

        # Register new vehicles
        for col in unused_cols:
            self._register_track(detections[col], current_time, w)

        return list(self.tracks.values())

    def _register_track(self, det: VehicleDetection, current_time: float, frame_width: int):
        track = TrackedVehicle(self.next_track_id, det, current_time)
        lane_width = frame_width / 4.0
        track.lane_id = min(4, max(1, int(det.center[0] // lane_width) + 1))
        # Initial default speed estimation based on vehicle class
        initial_speeds = {"car": 58.0, "bus": 44.0, "truck": 40.0, "motorcycle": 62.0, "bicycle": 20.0}
        track.speed_kmh = initial_speeds.get(track.class_name, 50.0)
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def reset_counts(self):
        self.total_in = 0
        self.total_out = 0
        self.class_counts = {k: 0 for k in self.class_counts}
        self.lane_counts = {k: 0 for k in self.lane_counts}
