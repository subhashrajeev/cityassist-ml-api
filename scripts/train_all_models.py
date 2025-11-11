"""
Script to train all ML models for CityAssist
Run this before deployment to generate all model artifacts
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.aqi_model import AQIAlertModel, generate_synthetic_aqi_data
from models.traffic_model import TrafficETAModel, generate_synthetic_traffic_data
from models.outage_model import OutageETAModel, generate_synthetic_outage_data
from models.image_classifier import CivicImageClassifier, generate_synthetic_image_data

def train_all_models():
    """Train all models and save artifacts"""

    # Create models directory
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("CityAssist Model Training Pipeline")
    print("=" * 60)

    # 1. Train AQI Alert Model
    print("\n[1/4] Training AQI Alert Model...")
    X_aqi, y_aqi = generate_synthetic_aqi_data(2000)
    aqi_model = AQIAlertModel()
    metrics = aqi_model.train(X_aqi, y_aqi)
    aqi_model.save("models/aqi_alert_model.pkl", "models/aqi_alert_scaler.pkl")
    print(f"   ✓ AQI Model trained successfully!")
    print(f"   Accuracy: {metrics['train_accuracy']:.4f}")

    # 2. Train Traffic ETA Model
    print("\n[2/4] Training Traffic ETA Model...")
    X_traffic, y_traffic = generate_synthetic_traffic_data(3000)
    traffic_model = TrafficETAModel()
    metrics = traffic_model.train(X_traffic, y_traffic)
    traffic_model.save("models/traffic_eta_model.pkl", "models/traffic_eta_scaler.pkl")
    print(f"   ✓ Traffic Model trained successfully!")
    print(f"   R² Score: {metrics['train_r2_score']:.4f}, MAE: {metrics['train_mae_minutes']:.2f} min")

    # 3. Train Outage ETA Model
    print("\n[3/4] Training Outage ETA Model...")
    X_outage, y_outage = generate_synthetic_outage_data(2000)
    outage_model = OutageETAModel()
    metrics = outage_model.train(X_outage, y_outage)
    outage_model.save("models/outage_eta_model.pkl", "models/outage_eta_scaler.pkl")
    print(f"   ✓ Outage Model trained successfully!")
    print(f"   R² Score: {metrics['train_r2_score']:.4f}, MAE: {metrics['train_mae_hours']:.2f} hours")

    # 4. Train Image Classifier
    print("\n[4/4] Training Image Classification Model...")
    print("   This may take several minutes...")
    X_img, y_img = generate_synthetic_image_data(1000)
    X_val, y_val = generate_synthetic_image_data(200)
    image_model = CivicImageClassifier()
    metrics = image_model.train(X_img, y_img, X_val, y_val, epochs=5)
    image_model.save("models/image_classifier.h5")
    print(f"   ✓ Image Model trained successfully!")
    print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}, Val Accuracy: {metrics['val_accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("✓ All models trained and saved successfully!")
    print("=" * 60)
    print("\nModel artifacts saved in the 'models/' directory:")
    print("  - aqi_alert_model.pkl & aqi_alert_scaler.pkl")
    print("  - traffic_eta_model.pkl & traffic_eta_scaler.pkl")
    print("  - outage_eta_model.pkl")
    print("  - image_classifier.h5")
    print("\nYou can now run the API with: python api/main.py")
    print("Or build the Docker image with: docker-compose up --build")

if __name__ == "__main__":
    train_all_models()
