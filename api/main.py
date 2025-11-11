"""
CityAssist Data Science API
FastAPI application for ML model inference
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import sys
import os
from pathlib import Path
import uuid
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.aqi_model import AQIAlertModel
from models.traffic_model import TrafficETAModel
from models.outage_model import OutageETAModel
from models.image_classifier import CivicImageClassifier
from api.utils import setup_logger, format_prediction_response

# Initialize FastAPI app
app = FastAPI(
    title="CityAssist Data Science API",
    description="ML-powered predictions for smart city operations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logger
logger = setup_logger(__name__)

# Global model instances
aqi_model = None
traffic_model = None
outage_model = None
image_model = None

# Request/Response Models
class AQIAlertRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID")
    zone_id: str = Field(..., description="Zone identifier")
    timestamp: Optional[str] = Field(None, description="Timestamp (ISO format)")
    pm25: float = Field(..., description="PM2.5 level")
    pm10: float = Field(..., description="PM10 level")
    no2: float = Field(0, description="NO2 level")
    so2: float = Field(0, description="SO2 level")
    co: float = Field(0, description="CO level")
    o3: float = Field(0, description="O3 level")
    pm25_change_pct: Optional[float] = Field(0, description="PM2.5 change percentage")
    temperature: float = Field(25, description="Temperature in Celsius")
    humidity: float = Field(50, description="Humidity percentage")
    wind_speed: float = Field(10, description="Wind speed in km/h")
    user_age: int = Field(35, description="User age")
    has_health_condition: int = Field(0, description="Has health condition (0/1)")

class TrafficSegment(BaseModel):
    segment_id: str
    timestamp: Optional[str] = None
    temperature: float = 25
    rain_mm: float = 0
    visibility_km: float = 10
    historical_avg_speed: float = 40
    segment_length_km: float = 1.0
    num_traffic_lights: int = 0
    is_highway: int = 0
    incident_nearby: int = 0

class RouteETARequest(BaseModel):
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    segments: List[TrafficSegment] = Field(..., description="Route segments")
    timestamp: Optional[str] = Field(None, description="Request timestamp")

class OutageETARequest(BaseModel):
    outage_id: str = Field(..., description="Outage identifier")
    timestamp: str = Field(..., description="Outage timestamp")
    cause: str = Field(..., description="Outage cause")
    zone: str = Field(..., description="Zone identifier")
    temperature: float = Field(20, description="Temperature")
    wind_speed: float = Field(15, description="Wind speed")
    affected_customers: int = Field(100, description="Number of affected customers")
    historical_mean_restore_hours: float = Field(3, description="Historical mean restore time")

class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    prediction: Any
    confidence: float
    explanation: Dict[str, Any]
    metadata: Dict[str, Any]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load all models on startup"""
    global aqi_model, traffic_model, outage_model, image_model

    logger.info("Loading ML models...")

    try:
        # Load AQI model
        aqi_model = AQIAlertModel()
        if os.path.exists("models/aqi_alert_model.pkl"):
            aqi_model.load("models/aqi_alert_model.pkl", "models/aqi_alert_scaler.pkl")
            logger.info("AQI Alert Model loaded successfully")
        else:
            logger.warning("AQI model not found, training new model...")
            from models.aqi_model import generate_synthetic_aqi_data
            X, y = generate_synthetic_aqi_data(2000)
            aqi_model.train(X, y)
            os.makedirs("models", exist_ok=True)
            aqi_model.save("models/aqi_alert_model.pkl", "models/aqi_alert_scaler.pkl")

        # Load Traffic model
        traffic_model = TrafficETAModel()
        if os.path.exists("models/traffic_eta_model.pkl"):
            traffic_model.load("models/traffic_eta_model.pkl", "models/traffic_eta_scaler.pkl")
            logger.info("Traffic ETA Model loaded successfully")
        else:
            logger.warning("Traffic model not found, training new model...")
            from models.traffic_model import generate_synthetic_traffic_data
            X, y = generate_synthetic_traffic_data(2000)
            traffic_model.train(X, y)
            traffic_model.save("models/traffic_eta_model.pkl", "models/traffic_eta_scaler.pkl")

        # Load Outage model
        outage_model = OutageETAModel()
        if os.path.exists("models/outage_eta_model.pkl"):
            outage_model.load("models/outage_eta_model.pkl")
            logger.info("Outage ETA Model loaded successfully")
        else:
            logger.warning("Outage model not found, training new model...")
            from models.outage_model import generate_synthetic_outage_data
            X, y = generate_synthetic_outage_data(1500)
            outage_model.train(X, y)
            outage_model.save("models/outage_eta_model.pkl", "models/outage_eta_scaler.pkl")

        # Load Image model
        image_model = CivicImageClassifier()
        if os.path.exists("models/image_classifier.h5"):
            image_model.load("models/image_classifier.h5")
            logger.info("Image Classifier loaded successfully")
        else:
            logger.warning("Image model not found. Will train on first use.")

        logger.info("All models loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "aqi": aqi_model is not None,
            "traffic": traffic_model is not None,
            "outage": outage_model is not None,
            "image": image_model is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "CityAssist Data Science API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "aqi_alert": "/api/v1/predict/aqi-alert",
            "route_eta": "/api/v1/predict/route-eta",
            "outage_eta": "/api/v1/predict/outage-eta",
            "image_classify": "/api/v1/classify/report-image"
        }
    }

# AQI Alert Prediction
@app.post("/api/v1/predict/aqi-alert", response_model=PredictionResponse)
async def predict_aqi_alert(request: AQIAlertRequest):
    """
    Predict AQI alert level for a user/zone
    """
    try:
        prediction_id = str(uuid.uuid4())

        # Convert request to dict
        features = request.dict()

        # Predict
        alert_level, probability, explanation = aqi_model.predict(features)

        # Format response
        response = format_prediction_response(
            prediction_id=prediction_id,
            prediction={
                "alert_level": alert_level,
                "zone_id": request.zone_id
            },
            confidence=probability,
            explanation=explanation,
            model_version="aqi-v1.0"
        )

        logger.info(f"AQI prediction: {alert_level} for zone {request.zone_id}")

        return response

    except Exception as e:
        logger.error(f"Error in AQI prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Route ETA Prediction
@app.post("/api/v1/predict/route-eta")
async def predict_route_eta(request: RouteETARequest):
    """
    Predict route ETA considering traffic and conditions
    """
    try:
        prediction_id = str(uuid.uuid4())

        # Convert segments to dict format
        segments = [segment.dict() for segment in request.segments]

        # Predict route ETA
        eta_result = traffic_model.predict_route_eta(segments)

        response = {
            "prediction_id": prediction_id,
            "timestamp": datetime.utcnow().isoformat(),
            "origin": request.origin,
            "destination": request.destination,
            "eta": eta_result,
            "metadata": {
                "model_version": "traffic-v1.0"
            }
        }

        logger.info(f"Route ETA prediction: {eta_result['total_eta_minutes']:.2f} min")

        return response

    except Exception as e:
        logger.error(f"Error in route ETA prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Outage ETA Prediction
@app.post("/api/v1/predict/outage-eta", response_model=PredictionResponse)
async def predict_outage_eta(request: OutageETARequest):
    """
    Predict utility outage restoration time
    """
    try:
        prediction_id = str(uuid.uuid4())

        # Convert request to dict
        features = request.dict()

        # Predict
        restore_hours, lower_bound, upper_bound, explanation = outage_model.predict(features)

        # Format response
        response = format_prediction_response(
            prediction_id=prediction_id,
            prediction={
                "restore_hours": round(restore_hours, 2),
                "lower_bound_hours": round(lower_bound, 2),
                "upper_bound_hours": round(upper_bound, 2),
                "outage_id": request.outage_id
            },
            confidence=0.85,  # Model-specific confidence
            explanation=explanation,
            model_version="outage-v1.0"
        )

        logger.info(f"Outage ETA prediction: {restore_hours:.2f} hours")

        return response

    except Exception as e:
        logger.error(f"Error in outage ETA prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Image Classification
@app.post("/api/v1/classify/report-image")
async def classify_report_image(file: UploadFile = File(...)):
    """
    Classify civic report image (pothole, garbage, tree fall, etc.)
    """
    try:
        prediction_id = str(uuid.uuid4())

        # Read image bytes
        image_bytes = await file.read()

        # Predict
        label, confidence, priority, explanation = image_model.predict(image_bytes)

        response = {
            "prediction_id": prediction_id,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": {
                "label": label,
                "priority": priority
            },
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "metadata": {
                "model_version": "image-v1.0",
                "filename": file.filename
            }
        }

        logger.info(f"Image classification: {label} (confidence: {confidence:.2f})")

        return response

    except Exception as e:
        logger.error(f"Error in image classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
