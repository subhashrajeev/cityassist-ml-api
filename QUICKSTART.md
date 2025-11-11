# CityAssist DS API - Quick Start Guide

Get the API running in 5 minutes!

## For DevOps Team: Fastest Deployment

### Option 1: Docker (Recommended)

```bash
# Clone and deploy in one command
git clone <your-repo-url>
cd cityassist-ds
docker-compose up --build -d

# Verify
curl http://localhost:8000/health
```

That's it! API is running on port 8000.

### Option 2: Direct Python

```bash
# Clone repository
git clone <your-repo-url>
cd cityassist-ds

# Install dependencies
pip install -r requirements.txt

# Start (models will train automatically on first run)
python start.py
```

## Quick Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {"aqi": true, "traffic": true, "outage": true, "image": true}
}
```

## Quick API Test

### Test AQI Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict/aqi-alert \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## API Endpoints

- **Health**: `GET /health`
- **AQI Alert**: `POST /api/v1/predict/aqi-alert`
- **Route ETA**: `POST /api/v1/predict/route-eta`
- **Outage ETA**: `POST /api/v1/predict/outage-eta`
- **Image Classify**: `POST /api/v1/classify/report-image`

## Interactive API Documentation

Once running, visit: `http://localhost:8000/docs`

## Troubleshooting

### Models not loading?
The API trains models automatically on first startup. Wait 2-3 minutes on first run.

Or manually train:
```bash
python scripts/train_all_models.py
```

### Port 8000 in use?
Edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Change to 8080
```

### Out of memory?
Increase Docker memory:
- Docker Desktop → Settings → Resources → Memory → 4GB

## Next Steps

- **Full Documentation**: See [README.md](README.md)
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Run Tests**: `python scripts/test_api.py`

## Support

Issues? Check logs:
```bash
# Docker
docker-compose logs -f

# Direct Python
# Check console output
```

---

**Ready to integrate?** The API is production-ready and follows REST best practices. All endpoints return JSON with proper error handling.
