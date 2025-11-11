# CityAssist Data Science API

Production-ready ML inference API for the CityAssist Smart City platform. Provides AI-powered predictions for air quality alerts, traffic/route ETA, utility outage restoration time, and civic report image classification.

## Features

- **AQI Alert Prediction**: Personalized air quality alerts based on pollutant levels and user profiles
- **Traffic/Route ETA**: Real-time traffic delay and route ETA predictions
- **Outage ETA Estimation**: Utility outage restoration time predictions
- **Image Classification**: Automatic classification of civic reports (pothole, garbage, tree fall, etc.)

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cityassist-ds
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train all models:
```bash
python scripts/train_all_models.py
```

This will generate all model artifacts in the `models/` directory.

### Running the API

#### Option 1: Direct Python

```bash
python api/main.py
```

or

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Option 2: Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build Docker image manually
docker build -t cityassist-ds-api .
docker run -p 8000:8000 cityassist-ds-api
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```bash
GET /health
```

Returns health status and loaded models.

### 1. AQI Alert Prediction

```bash
POST /api/v1/predict/aqi-alert
```

**Request Body:**
```json
{
  "zone_id": "Zone-A",
  "pm25": 180,
  "pm10": 250,
  "no2": 45,
  "so2": 25,
  "co": 1.5,
  "o3": 50,
  "pm25_change_pct": 60,
  "temperature": 28,
  "humidity": 65,
  "wind_speed": 8,
  "user_age": 35,
  "has_health_condition": 1
}
```

**Response:**
```json
{
  "prediction_id": "uuid",
  "timestamp": "2024-11-11T10:00:00Z",
  "prediction": {
    "alert_level": "HIGH",
    "zone_id": "Zone-A"
  },
  "confidence": 0.87,
  "explanation": {
    "reason": "PM2.5 level elevated at 180.0",
    "key_factors": [...]
  },
  "metadata": {
    "model_version": "aqi-v1.0"
  }
}
```

### 2. Route ETA Prediction

```bash
POST /api/v1/predict/route-eta
```

**Request Body:**
```json
{
  "origin": "Location A",
  "destination": "Location B",
  "segments": [
    {
      "segment_id": "seg-1",
      "temperature": 22,
      "rain_mm": 8,
      "visibility_km": 5,
      "historical_avg_speed": 30,
      "segment_length_km": 5,
      "num_traffic_lights": 6,
      "is_highway": 0,
      "incident_nearby": 1
    }
  ]
}
```

**Response:**
```json
{
  "prediction_id": "uuid",
  "timestamp": "2024-11-11T10:00:00Z",
  "origin": "Location A",
  "destination": "Location B",
  "eta": {
    "total_eta_minutes": 45.5,
    "total_delay_minutes": 15.3,
    "total_distance_km": 5.0,
    "segments": [...]
  },
  "metadata": {
    "model_version": "traffic-v1.0"
  }
}
```

### 3. Outage ETA Prediction

```bash
POST /api/v1/predict/outage-eta
```

**Request Body:**
```json
{
  "outage_id": "outage-123",
  "timestamp": "2024-11-11T14:30:00",
  "cause": "storm_damage",
  "zone": "zone_C",
  "temperature": 15,
  "wind_speed": 55,
  "affected_customers": 2500,
  "historical_mean_restore_hours": 5.5
}
```

**Response:**
```json
{
  "prediction_id": "uuid",
  "timestamp": "2024-11-11T14:30:00Z",
  "prediction": {
    "restore_hours": 8.5,
    "lower_bound_hours": 5.95,
    "upper_bound_hours": 11.05,
    "outage_id": "outage-123"
  },
  "confidence": 0.85,
  "explanation": {
    "reason": "Extended restoration time due to: storm damage requires extensive repairs",
    "estimated_restoration": "Approximately 8h 30m"
  },
  "metadata": {
    "model_version": "outage-v1.0"
  }
}
```

### 4. Image Classification

```bash
POST /api/v1/classify/report-image
Content-Type: multipart/form-data
```

**Request:**
- file: Image file (JPEG, PNG)

**Response:**
```json
{
  "prediction_id": "uuid",
  "timestamp": "2024-11-11T10:00:00Z",
  "prediction": {
    "label": "pothole",
    "priority": "high"
  },
  "confidence": 0.91,
  "explanation": {
    "primary_class": "pothole",
    "confidence_level": "high",
    "requires_review": false,
    "alternative_classes": []
  },
  "metadata": {
    "model_version": "image-v1.0",
    "filename": "report.jpg"
  }
}
```

## Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# AQI Alert
curl -X POST http://localhost:8000/api/v1/predict/aqi-alert \
  -H "Content-Type: application/json" \
  -d '{"zone_id":"Zone-A","pm25":180,"pm10":250,"no2":45,"so2":25,"co":1.5,"o3":50,"temperature":28,"humidity":65,"wind_speed":8,"user_age":35,"has_health_condition":1}'

# Image Classification
curl -X POST http://localhost:8000/api/v1/classify/report-image \
  -F "file=@image.jpg"
```

### Using Python

```python
import requests

# AQI Alert
response = requests.post(
    "http://localhost:8000/api/v1/predict/aqi-alert",
    json={
        "zone_id": "Zone-A",
        "pm25": 180,
        "pm10": 250,
        "no2": 45,
        "so2": 25,
        "co": 1.5,
        "o3": 50,
        "temperature": 28,
        "humidity": 65,
        "wind_speed": 8,
        "user_age": 35,
        "has_health_condition": 1
    }
)
print(response.json())
```

## Project Structure

```
cityassist-ds/
├── api/
│   ├── main.py              # FastAPI application
│   └── utils.py             # Utility functions
├── models/
│   ├── aqi_model.py         # AQI alert prediction model
│   ├── traffic_model.py     # Traffic/route ETA model
│   ├── outage_model.py      # Outage ETA estimation model
│   └── image_classifier.py  # Image classification model
├── scripts/
│   └── train_all_models.py  # Model training script
├── config/
│   └── config.yaml          # Configuration file
├── tests/
│   └── test_api.py          # API tests
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Model Details

### 1. AQI Alert Model
- **Algorithm**: Gradient Boosting Classifier
- **Features**: PM2.5, PM10, NO2, SO2, CO, O3, temporal features, weather, user profile
- **Output**: Alert level (LOW, MODERATE, HIGH, SEVERE) with confidence and explanation

### 2. Traffic ETA Model
- **Algorithm**: Random Forest Regressor
- **Features**: Time of day, weather, historical speeds, road characteristics, incidents
- **Output**: Delay in minutes with confidence and reason

### 3. Outage ETA Model
- **Algorithm**: Gradient Boosting Regressor
- **Features**: Outage cause, zone, weather, affected customers, historical data
- **Output**: Restoration time in hours with prediction intervals

### 4. Image Classifier
- **Architecture**: MobileNetV2 (transfer learning)
- **Classes**: pothole, garbage, tree_fall, streetlight, other
- **Output**: Label, confidence, priority, and explainability

## Deployment

### Production Deployment Checklist

1. **Environment Variables**
   - Set `LOG_LEVEL=INFO` for production
   - Configure proper CORS origins in `api/main.py`

2. **Model Artifacts**
   - Ensure all model files are present in `models/` directory
   - Consider using cloud storage (S3, GCS) for large models

3. **Scaling**
   - Use `--workers` parameter for uvicorn to scale horizontally
   - Deploy behind a load balancer (nginx, HAProxy)
   - Consider Kubernetes for orchestration

4. **Monitoring**
   - Implement Prometheus metrics collection
   - Set up Grafana dashboards
   - Configure alerting for model performance degradation

5. **Security**
   - Implement authentication (JWT, API keys)
   - Use HTTPS in production
   - Validate and sanitize all inputs

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cityassist-ds-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cityassist-ds-api
  template:
    metadata:
      labels:
        app: cityassist-ds-api
    spec:
      containers:
      - name: api
        image: cityassist-ds-api:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: cityassist-ds-api-service
spec:
  selector:
    app: cityassist-ds-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black api/ models/
flake8 api/ models/
```

## Troubleshooting

### Models not loading
- Ensure you've run `python scripts/train_all_models.py`
- Check that `models/` directory contains all .pkl and .h5 files

### Memory issues
- Reduce batch size in model training
- Use smaller model variants
- Increase container memory limits

### Slow inference
- Use GPU for image classification
- Enable model quantization
- Implement request batching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## Support

For issues and questions:
- Create an issue on GitHub
- Contact: [your-email@example.com]

## Acknowledgments

- CityAssist Hackathon Team
- Dataset providers (Kaggle, IMD, NOAA)
- Open source community

---

**Built with** Python, FastAPI, scikit-learn, TensorFlow, Docker
