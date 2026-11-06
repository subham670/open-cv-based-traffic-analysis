"""
FlowPulse ITMS - Database Models & Persistence Layer
Stores telemetry snapshots, incident logs, speed violations, and historical aggregations.
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

os.makedirs("backend/data", exist_ok=True)
DATABASE_URL = "sqlite:///backend/data/traffic_analytics.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TelemetrySnapshot(Base):
    __tablename__ = "telemetry_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String, index=True)
    total_in = Column(Integer, default=0)
    total_out = Column(Integer, default=0)
    active_vehicles = Column(Integer, default=0)
    avg_speed = Column(Float, default=0.0)
    congestion_index = Column(Float, default=0.0)
    los_level = Column(String, default="A")
    density_ratio = Column(Float, default=0.0)
    class_breakdown = Column(JSON, default=dict)
    lane_breakdown = Column(JSON, default=dict)

class SpeedViolation(Base):
    __tablename__ = "speed_violations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String)
    track_id = Column(Integer)
    vehicle_class = Column(String)
    recorded_speed = Column(Float)
    speed_limit = Column(Float)
    lane_id = Column(Integer)
    direction = Column(String)

class TrafficIncident(Base):
    __tablename__ = "traffic_incidents"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String)
    incident_type = Column(String)  # STALLED_VEHICLE, WRONG_WAY, SEVERE_CONGESTION, SPEED_VIOLATION
    severity = Column(String)       # LOW, MEDIUM, HIGH, CRITICAL
    description = Column(String)
    track_id = Column(Integer, nullable=True)
    lane_id = Column(Integer, nullable=True)
    resolved = Column(Boolean, default=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
