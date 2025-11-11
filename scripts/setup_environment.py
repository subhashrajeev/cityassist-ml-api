"""
Setup script to prepare environment and train models
This script can be run independently or as part of deployment
"""
import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("ERROR: Python 3.10+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n Installing dependencies...")
    packages = [
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "python-multipart",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "tensorflow",
        "Pillow",
        "pyyaml",
        "requests"
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])

    print("✓ Dependencies installed")

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    dirs = ["models", "data", "logs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Directories created")

def train_models():
    """Train all models"""
    print("\nTraining models...")
    print("This may take several minutes...")

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from models.aqi_model import AQIAlertModel, generate_synthetic_aqi_data
        from models.traffic_model import TrafficETAModel, generate_synthetic_traffic_data
        from models.outage_model import OutageETAModel, generate_synthetic_outage_data

        # AQI Model
        print("\n[1/3] Training AQI Model...")
        X, y = generate_synthetic_aqi_data(2000)
        model = AQIAlertModel()
        model.train(X, y)
        model.save("models/aqi_alert_model.pkl", "models/aqi_alert_scaler.pkl")
        print("✓ AQI Model trained")

        # Traffic Model
        print("\n[2/3] Training Traffic Model...")
        X, y = generate_synthetic_traffic_data(2000)
        model = TrafficETAModel()
        model.train(X, y)
        model.save("models/traffic_eta_model.pkl", "models/traffic_eta_scaler.pkl")
        print("✓ Traffic Model trained")

        # Outage Model
        print("\n[3/3] Training Outage Model...")
        X, y = generate_synthetic_outage_data(1500)
        model = OutageETAModel()
        model.train(X, y)
        model.save("models/outage_eta_model.pkl", "models/outage_eta_scaler.pkl")
        print("✓ Outage Model trained")

        print("\n✓ All models trained successfully!")
        return True

    except Exception as e:
        print(f"ERROR training models: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("CityAssist DS - Environment Setup")
    print("="*60)

    if not check_python_version():
        sys.exit(1)

    create_directories()

    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train models: python scripts/train_all_models.py")
    print("3. Start API: python api/main.py")

if __name__ == "__main__":
    main()
