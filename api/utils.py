"""
Utility functions for CityAssist DS API
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any
import yaml
from pathlib import Path

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Set up structured JSON logging"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_prediction_response(
    prediction_id: str,
    prediction: Any,
    confidence: float,
    explanation: Dict[str, Any],
    model_version: str
) -> Dict[str, Any]:
    """Format standardized prediction response"""
    return {
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "explanation": explanation,
        "metadata": {
            "model_version": model_version
        }
    }

def validate_input(data: Dict[str, Any], required_fields: list) -> tuple:
    """Validate input data has required fields"""
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, "Valid"

logger = setup_logger(__name__)
