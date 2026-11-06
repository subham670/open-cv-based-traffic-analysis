"""
FlowPulse ITMS - Deep Learning & Computer Vision Vehicle Detector
Combines real-time NanoDet ONNX Deep Learning detection with robust
Computer Vision MOG2 adaptive background-subtraction & morphological contour filtering.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from backend.models.nanodet_helper import NanoDet

logger = logging.getLogger("flowpulse.detector")

COCO_VEHICLE_MAP = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class VehicleDetection:
    def __init__(self, box: Tuple[int, int, int, int], class_name: str, confidence: float):
        self.box = box
        self.x, self.y, self.w, self.h = box
        self.class_name = class_name
        self.confidence = float(confidence)
        self.center = (int(self.x + self.w / 2), int(self.y + self.h / 2))
        self.area = self.w * self.h

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [self.x, self.y, self.w, self.h],
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "center": self.center,
            "area": self.area
        }

class VehicleDetector:
    def __init__(
        self,
        onnx_model_path: str = "models/nanodet.onnx",
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.50,
        target_classes: List[str] = None
    ):
        self.onnx_model_path = onnx_model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes or ["car", "bus", "truck", "motorcycle", "bicycle"]
        
        self.dl_model: NanoDet = None
        self.is_dnn_ready = False

        # Computer Vision Fallback (Adaptive Background Subtraction + Morphological filtering)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        self._init_network()

    def _init_network(self):
        if os.path.exists(self.onnx_model_path):
            try:
                self.dl_model = NanoDet(
                    self.onnx_model_path,
                    prob_threshold=self.conf_threshold,
                    iou_threshold=self.nms_threshold
                )
                self.is_dnn_ready = True
                logger.info("NanoDet Deep Learning ONNX engine initialized successfully.")
            except Exception as e:
                logger.warning(f"ONNX DNN initialization error: {e}. Falling back to Computer Vision engine.")
                self.is_dnn_ready = False
        else:
            logger.info("ONNX model weights not found. Using Computer Vision background engine.")

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        if not self.is_dnn_ready:
            self._init_network()

        # Try Deep Learning detector first
        if self.is_dnn_ready and self.dl_model is not None:
            try:
                detections = self._detect_dnn(frame)
                if len(detections) > 0:
                    return detections
            except Exception as e:
                logger.debug(f"DNN inference notice: {e}, falling back to CV engine")

        # Computer Vision fallback / augmentation
        return self._detect_cv_fallback(frame)

    def _detect_dnn(self, frame: np.ndarray) -> List[VehicleDetection]:
        h, w = frame.shape[:2]
        # Resize frame to 416x416 for NanoDet
        resized = cv2.resize(frame, (416, 416))
        preds = self.dl_model.infer(resized)

        detections: List[VehicleDetection] = []
        if preds is None or len(preds) == 0:
            return detections

        scale_x = w / 416.0
        scale_y = h / 416.0

        for pred in preds:
            x1, y1, x2, y2, conf, cls_id = pred
            cls_id = int(cls_id)
            if cls_id in COCO_VEHICLE_MAP:
                class_name = COCO_VEHICLE_MAP[cls_id]
                if class_name in self.target_classes and conf >= self.conf_threshold:
                    rx1 = int(max(0, x1 * scale_x))
                    ry1 = int(max(0, y1 * scale_y))
                    rx2 = int(min(w, x2 * scale_x))
                    ry2 = int(min(h, y2 * scale_y))
                    bw = rx2 - rx1
                    bh = ry2 - ry1
                    if bw > 15 and bh > 15:
                        detections.append(VehicleDetection((rx1, ry1, bw, bh), class_name, float(conf)))

        return detections

    def _detect_cv_fallback(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Robust Computer Vision vehicle detector using adaptive MOG2 background modeling,
        morphological opening/closing, and vehicle geometry heuristics (area, aspect ratio).
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blur)
        
        # Morphological filtering to eliminate noise and connect vehicle components
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        dilated = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[VehicleDetection] = []

        min_area = int(0.001 * w * h)  # Min vehicle size
        max_area = int(0.35 * w * h)   # Max vehicle size

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh)

                # Heuristic Vehicle Classification based on dimensions & perspective
                if area > 0.04 * w * h:
                    c_name = "truck" if aspect_ratio > 1.2 else "bus"
                elif area > 0.008 * w * h:
                    c_name = "car"
                elif aspect_ratio < 0.65 or area < 0.005 * w * h:
                    c_name = "motorcycle"
                else:
                    c_name = "car"

                conf = min(0.95, 0.55 + (area / max_area) * 0.4)
                detections.append(VehicleDetection((x, y, bw, bh), c_name, conf))

        return detections
