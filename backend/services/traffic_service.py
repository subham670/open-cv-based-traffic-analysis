"""
FlowPulse ITMS - Traffic Service & Business Logic
Coordinates live telemetry persistence, database historical aggregation,
incident handling, and audit report generation.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import csv
import io

from backend.database import SessionLocal, TelemetrySnapshot, SpeedViolation, TrafficIncident
from backend.services.video_stream import VideoStreamProcessor
import logging

logger = logging.getLogger("flowpulse.service")

class TrafficService:
    def __init__(self, processor: VideoStreamProcessor):
        self.processor = processor
        self.last_db_persist_time = datetime.utcnow()
        self.logged_violations = set()
        self.logged_incidents = set()

    def tick_and_persist(self, force: bool = False):
        """
        Periodically saves snapshot to SQLite database
        """
        now = datetime.utcnow()
        telemetry = self.processor.latest_telemetry
        if force or (now - self.last_db_persist_time).total_seconds() >= 5:
            self.last_db_persist_time = now
            db = SessionLocal()
            try:
                snapshot = TelemetrySnapshot(
                    timestamp=now,
                    camera_id=telemetry.get("source_type", "cam_01"),
                    total_in=telemetry.get("total_in", 0),
                    total_out=telemetry.get("total_out", 0),
                    active_vehicles=telemetry.get("active_vehicles", 0),
                    avg_speed=telemetry.get("avg_speed_kmh", 0.0),
                    congestion_index=telemetry.get("congestion_index", 0.0),
                    los_level=telemetry.get("los_level", "A"),
                    density_ratio=telemetry.get("congestion_index", 0.0) / 100.0,
                    class_breakdown=telemetry.get("class_counts", {}),
                    lane_breakdown=telemetry.get("lane_counts", {})
                )
                db.add(snapshot)

                # Persist unlogged incidents
                for inc in telemetry.get("incidents", []):
                    inc_key = (inc["type"], inc.get("track_id"))
                    if inc_key not in self.logged_incidents:
                        self.logged_incidents.add(inc_key)
                        db_inc = TrafficIncident(
                            timestamp=now,
                            camera_id=telemetry.get("source_type", "cam_01"),
                            incident_type=inc["type"],
                            severity=inc["severity"],
                            description=inc["message"],
                            track_id=inc.get("track_id"),
                            lane_id=inc.get("lane_id"),
                            resolved=False
                        )
                        db.add(db_inc)

                db.commit()
            except Exception as e:
                logger.error(f"Error persisting telemetry: {e}")
                db.rollback()
            finally:
                db.close()

    def get_historical_telemetry(self, limit: int = 50) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            records = db.query(TelemetrySnapshot).order_by(TelemetrySnapshot.timestamp.desc()).limit(limit).all()
            return [
                {
                    "timestamp": r.timestamp.strftime("%H:%M:%S"),
                    "total_in": r.total_in,
                    "total_out": r.total_out,
                    "active_vehicles": r.active_vehicles,
                    "avg_speed": r.avg_speed,
                    "congestion_index": r.congestion_index,
                    "los_level": r.los_level
                }
                for r in reversed(records)
            ]
        finally:
            db.close()

    def get_incidents(self, limit: int = 20) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            incidents = db.query(TrafficIncident).order_by(TrafficIncident.timestamp.desc()).limit(limit).all()
            return [
                {
                    "id": i.id,
                    "timestamp": i.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": i.incident_type,
                    "severity": i.severity,
                    "description": i.description,
                    "track_id": i.track_id,
                    "lane_id": i.lane_id,
                    "resolved": i.resolved
                }
                for i in incidents
            ]
        finally:
            db.close()

    def resolve_incident(self, incident_id: int) -> bool:
        db = SessionLocal()
        try:
            inc = db.query(TrafficIncident).filter(TrafficIncident.id == incident_id).first()
            if inc:
                inc.resolved = True
                db.commit()
                return True
            return False
        finally:
            db.close()

    def generate_csv_export(self) -> str:
        db = SessionLocal()
        try:
            records = db.query(TelemetrySnapshot).order_by(TelemetrySnapshot.timestamp.desc()).limit(500).all()
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Timestamp", "Camera_ID", "Total_In", "Total_Out", "Active_Vehicles", "Avg_Speed_KMH", "Congestion_Index", "LOS_Level"])
            for r in records:
                writer.writerow([
                    r.timestamp.isoformat(),
                    r.camera_id,
                    r.total_in,
                    r.total_out,
                    r.active_vehicles,
                    r.avg_speed,
                    r.congestion_index,
                    r.los_level
                ])
            return output.getvalue()
        finally:
            db.close()
