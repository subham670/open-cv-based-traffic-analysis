"""
FlowPulse ITMS - High-Fidelity Realistic Traffic Scene Synthesizer
Generates realistic multi-lane perspective traffic feeds with multi-class vehicles,
shadows, headlights, realistic physics, lane changes, and varying density.
"""
import cv2
import numpy as np
import random
import time
from typing import List, Dict, Any

class SimVehicle:
    def __init__(self, lane: int, direction: int, v_class: str, y: float, speed: float, color: tuple):
        self.lane = lane              # 0, 1 (Northbound/Up) or 2, 3 (Southbound/Down)
        self.direction = direction    # -1 for Up, +1 for Down
        self.v_class = v_class        # 'car', 'bus', 'truck', 'motorcycle'
        self.y = y
        self.speed = speed
        self.target_speed = speed
        self.color = color
        self.lane_change_progress = 0.0
        self.target_lane = lane
        self.stalled_counter = 0

class TrafficSceneGenerator:
    def __init__(self, width: int = 960, height: int = 540, mode: str = "highway"):
        self.width = width
        self.height = height
        self.mode = mode
        self.vehicles: List[SimVehicle] = []
        self.frame_count = 0
        self.last_spawn_time = 0

        # Lane center X coordinates at bottom (perspective projection)
        # Highway layout:
        # Lanes 0, 1: Southbound (traffic going down, towards camera)
        # Lanes 2, 3: Northbound (traffic going up, away from camera)
        self.bottom_lanes = [width * 0.22, width * 0.40, width * 0.60, width * 0.78]
        self.top_lanes = [width * 0.38, width * 0.46, width * 0.54, width * 0.62]

        self.colors = [
            (35, 35, 190),   # Crimson Red
            (210, 210, 210), # Metallic Silver
            (25, 25, 25),    # Obsidian Black
            (180, 120, 20),  # Electric Blue
            (20, 150, 210),  # Solar Gold
            (60, 60, 60)     # Carbon Gray
        ]

        # Seed initial vehicles
        for _ in range(8):
            self._spawn_vehicle(random.uniform(50, height - 100))

    def _spawn_vehicle(self, y_pos: float = None):
        lane = random.choice([0, 1, 2, 3])
        direction = 1 if lane < 2 else -1  # 0, 1 go down; 2, 3 go up

        v_type_roll = random.random()
        if v_type_roll < 0.60:
            v_class = "car"
            base_speed = random.uniform(3.5, 5.5)
        elif v_type_roll < 0.80:
            v_class = "truck"
            base_speed = random.uniform(2.5, 3.8)
        elif v_type_roll < 0.90:
            v_class = "bus"
            base_speed = random.uniform(2.8, 4.0)
        else:
            v_class = "motorcycle"
            base_speed = random.uniform(4.0, 6.5)

        if y_pos is None:
            y = 10.0 if direction == 1 else self.height - 20.0
        else:
            y = y_pos

        color = random.choice(self.colors)
        veh = SimVehicle(lane, direction, v_class, y, base_speed, color)
        
        # Occasionally inject a stalled vehicle for incident detection demo
        if random.random() < 0.05 and self.frame_count > 300:
            veh.stalled_counter = 120

        self.vehicles.append(veh)

    def generate_frame(self) -> np.ndarray:
        self.frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Render Road & Surroundings (Asphalt + Horizon + Grass/Shoulders)
        # Background landscape
        frame[:] = (45, 55, 45) # Subtle grass/curb dark green

        # Perspective Road Polygon
        road_pts = np.array([
            [int(self.top_lanes[0] - 60), 0],
            [int(self.top_lanes[3] + 60), 0],
            [int(self.bottom_lanes[3] + 90), self.height],
            [int(self.bottom_lanes[0] - 90), self.height]
        ], np.int32)
        cv2.fillPoly(frame, [road_pts], (42, 42, 44)) # Smooth asphalt

        # Yellow Shoulder Lines
        cv2.line(frame, (int(self.top_lanes[0] - 50), 0), (int(self.bottom_lanes[0] - 70), self.height), (30, 200, 230), 4)
        cv2.line(frame, (int(self.top_lanes[3] + 50), 0), (int(self.bottom_lanes[3] + 70), self.height), (30, 200, 230), 4)

        # Center Median Barrier
        med_top = int((self.top_lanes[1] + self.top_lanes[2]) / 2)
        med_bot = int((self.bottom_lanes[1] + self.bottom_lanes[2]) / 2)
        cv2.line(frame, (med_top, 0), (med_bot, self.height), (120, 120, 130), 6)
        cv2.line(frame, (med_top - 4, 0), (med_bot - 4, self.height), (40, 215, 235), 2)

        # Dashed Lane Dividers (with animated road texture motion)
        offset = (self.frame_count * 4) % 40
        for l_idx in [0, 2]:
            x_top = int((self.top_lanes[l_idx] + self.top_lanes[l_idx + 1]) / 2)
            x_bot = int((self.bottom_lanes[l_idx] + self.bottom_lanes[l_idx + 1]) / 2)
            for y_seg in range(0, self.height, 35):
                y_start = (y_seg + offset) % self.height
                y_end = min(self.height, y_start + 18)
                # Linear interpolation for perspective X
                t1 = y_start / self.height
                t2 = y_end / self.height
                x1 = int(x_top + t1 * (x_bot - x_top))
                x2 = int(x_top + t2 * (x_bot - x_top))
                cv2.line(frame, (x1, y_start), (x2, y_end), (220, 220, 220), 2)

        # Spawn new vehicles if count is below threshold
        if len(self.vehicles) < 11 and random.random() < 0.08:
            self._spawn_vehicle()

        # Sort vehicles by Y for realistic depth occlusion (distant drawn first)
        self.vehicles.sort(key=lambda v: v.y)

        # 2. Update & Render Vehicles
        surviving_vehicles = []
        for v in self.vehicles:
            # Perspective scaling factor (0.45 at horizon, 1.25 at bottom)
            depth_scale = 0.45 + (v.y / self.height) * 0.80

            # Base dimensions by vehicle class
            if v.v_class == "car":
                bw = int(48 * depth_scale)
                bh = int(82 * depth_scale)
            elif v.v_class == "truck":
                bw = int(62 * depth_scale)
                bh = int(140 * depth_scale)
            elif v.v_class == "bus":
                bw = int(60 * depth_scale)
                bh = int(130 * depth_scale)
            else:  # motorcycle
                bw = int(24 * depth_scale)
                bh = int(46 * depth_scale)

            # Perspective X position along lane
            t = max(0.0, min(1.0, v.y / self.height))
            top_x = self.top_lanes[v.lane]
            bot_x = self.bottom_lanes[v.lane]
            cx = int(top_x + t * (bot_x - top_x))
            cy = int(v.y)

            # Vehicle movement physics
            if v.stalled_counter > 0:
                v.stalled_counter -= 1
                v.speed = 0.0
            else:
                v.speed = v.target_speed

            v.y += v.direction * v.speed

            # Vehicle bounding box coordinates
            vx = int(cx - bw / 2)
            vy = int(cy - bh / 2)

            # Draw Ground Shadow
            shadow_poly = np.array([
                [vx - 4, vy + bh],
                [vx + bw + 4, vy + bh],
                [vx + bw + 10, vy + bh + 12],
                [vx - 10, vy + bh + 12]
            ], np.int32)
            cv2.fillPoly(frame, [shadow_poly], (25, 25, 26))

            # Draw Vehicle Body
            cv2.rectangle(frame, (vx, vy), (vx + bw, vy + bh), v.color, -1)
            cv2.rectangle(frame, (vx, vy), (vx + bw, vy + bh), (30, 30, 30), 2)

            # Roof & Windshield
            win_h = max(4, int(bh * 0.20))
            if v.direction == 1:
                # Windshield facing camera (at bottom of vehicle top-view)
                cv2.rectangle(frame, (vx + 3, vy + int(bh * 0.55)), (vx + bw - 3, vy + int(bh * 0.55) + win_h), (80, 110, 140), -1)
                # Headlights
                cv2.circle(frame, (vx + 5, vy + bh - 2), 3, (180, 240, 255), -1)
                cv2.circle(frame, (vx + bw - 5, vy + bh - 2), 3, (180, 240, 255), -1)
            else:
                # Windshield facing away (at top)
                cv2.rectangle(frame, (vx + 3, vy + int(bh * 0.25)), (vx + bw - 3, vy + int(bh * 0.25) + win_h), (80, 110, 140), -1)
                # Taillights (Red)
                cv2.circle(frame, (vx + 5, vy + bh - 2), 3, (30, 30, 220), -1)
                cv2.circle(frame, (vx + bw - 5, vy + bh - 2), 3, (30, 30, 220), -1)

            # Check boundary conditions
            if -100 <= v.y <= self.height + 100:
                surviving_vehicles.append(v)

        self.vehicles = surviving_vehicles
        return frame
