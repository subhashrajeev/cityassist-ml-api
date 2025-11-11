"""
Quick start script for CityAssist DS API
Trains models if needed and starts the API
"""
import os
import sys
import subprocess

def check_models_exist():
    """Check if model files exist"""
    required_models = [
        "models/aqi_alert_model.pkl",
        "models/traffic_eta_model.pkl",
        "models/outage_eta_model.pkl"
    ]

    all_exist = all(os.path.exists(model) for model in required_models)
    return all_exist

def train_models():
    """Train models using script"""
    print("\nModel artifacts not found. Training models...")
    print("This will take a few minutes...")
    subprocess.run([sys.executable, "scripts/train_all_models.py"])

def start_api():
    """Start the FastAPI server"""
    print("\nStarting CityAssist DS API...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def main():
    """Main function"""
    print("="*60)
    print("CityAssist Data Science API - Quick Start")
    print("="*60)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Check if models exist, train if needed
    if not check_models_exist():
        print("\nℹ Models will be trained automatically on first API start")
        print("The API will train models in the background...")

    # Start API
    start_api()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down API...")
        sys.exit(0)
