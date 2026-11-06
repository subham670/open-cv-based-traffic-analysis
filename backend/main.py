"""
FlowPulse ITMS - Production FastAPI Server
Provides high-frequency real-time telemetry, MJPEG video streaming,
predictive traffic forecasting, and camera management endpoints.
"""
import os
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.config import settings
from backend.database import init_db
from backend.services.video_stream import VideoStreamProcessor
from backend.services.traffic_service import TrafficService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("flowpulse.api")

# Ensure required folders exist
os.makedirs("backend/data", exist_ok=True)
os.makedirs("backend/uploads", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# Initialize Database Schema
init_db()

# Core Services
processor = VideoStreamProcessor()
traffic_service = TrafficService(processor)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade Computer Vision Urban Traffic Management & Analytics Engine"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background Persistence Task
async def persistence_worker():
    while True:
        try:
            traffic_service.tick_and_persist()
        except Exception as e:
            logger.error(f"Persistence worker error: {e}")
        await asyncio.sleep(4)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(persistence_worker())
    logger.info("FlowPulse ITMS API Engine started successfully.")

# Schemas
class SourceSwitchRequest(BaseModel):
    source_type: str
    file_path: Optional[str] = ""
    webcam_idx: Optional[int] = 0

class DisplayOptionsRequest(BaseModel):
    show_boxes: Optional[bool] = None
    show_trails: Optional[bool] = None
    show_tripwires: Optional[bool] = None
    show_speed: Optional[bool] = None
    show_hud: Optional[bool] = None
    show_lanes: Optional[bool] = None

class TripwireRequest(BaseModel):
    line_in_y: int
    line_out_y: int

# Endpoints
@app.get("/api/health")
def get_health():
    return {
        "status": "ONLINE",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "active_source": processor.current_source_type,
        "dnn_ready": processor.detector.is_dnn_ready
    }

@app.get("/api/stream")
def stream_video():
    """
    High-performance MJPEG live stream of annotated computer-vision frames
    """
    return StreamingResponse(
        processor.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/telemetry")
def get_telemetry():
    """
    Returns latest frame analytics, vehicle counts, Level of Service, speeds & incidents
    """
    return {
        "success": True,
        "telemetry": processor.latest_telemetry,
        "display_options": {
            "show_boxes": processor.show_boxes,
            "show_trails": processor.show_trails,
            "show_tripwires": processor.show_tripwires,
            "show_speed": processor.show_speed,
            "show_hud": processor.show_hud,
            "show_lanes": processor.show_lanes
        },
        "tripwires": {
            "line_in_y": processor.tracker.line_in_y,
            "line_out_y": processor.tracker.line_out_y
        }
    }

@app.get("/api/forecast")
def get_forecast():
    """
    Returns 24-hour predictive traffic forecast, peak congestion windows and recommended actions
    """
    current_cong = processor.latest_telemetry.get("congestion_index", 45.0)
    forecast = processor.forecaster.generate_24h_forecast(current_cong)
    return {"success": True, "forecast": forecast}

@app.get("/api/history")
def get_history(limit: int = 40):
    """
    Returns historical telemetry snapshots for charting
    """
    return {"success": True, "history": traffic_service.get_historical_telemetry(limit)}

@app.get("/api/incidents")
def get_incidents(limit: int = 25):
    """
    Returns live and logged traffic incidents
    """
    return {"success": True, "incidents": traffic_service.get_incidents(limit)}

@app.post("/api/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: int):
    success = traffic_service.resolve_incident(incident_id)
    if not success:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"success": True, "message": f"Incident #{incident_id} marked as resolved"}

@app.get("/api/cameras")
def get_cameras():
    return {"success": True, "cameras": settings.AVAILABLE_CAMERAS}

@app.post("/api/source")
def switch_source(req: SourceSwitchRequest):
    processor.switch_source(req.source_type, req.file_path, req.webcam_idx)
    traffic_service.tick_and_persist(force=True)
    return {"success": True, "active_source": req.source_type}

@app.post("/api/display")
def update_display(req: DisplayOptionsRequest):
    opts = {k: v for k, v in req.dict().items() if v is not None}
    processor.set_display_options(opts)
    return {"success": True, "display_options": opts}

@app.post("/api/tripwires")
def update_tripwires(req: TripwireRequest):
    processor.set_tripwires(req.line_in_y, req.line_out_y)
    return {"success": True, "tripwires": req.dict()}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload user's traffic video (MP4, AVI, MOV) for live Computer Vision analysis
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    dest_path = os.path.join("backend/uploads", file.filename)
    with open(dest_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    processor.switch_source("custom_file", dest_path)
    return {
        "success": True,
        "message": f"Uploaded and activated: {file.filename}",
        "file_path": dest_path
    }

@app.get("/api/export")
def export_csv():
    csv_data = traffic_service.generate_csv_export()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_telemetry_export.csv"}
    )

# Static Frontend mounting
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=False)
