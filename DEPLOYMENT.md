# CityAssist DS API - Deployment Guide for DevOps

This document provides step-by-step instructions for deploying the CityAssist Data Science API.

## Pre-Deployment Checklist

- [ ] All model artifacts are present in `models/` directory
- [ ] Docker is installed and running
- [ ] Required ports are available (8000 by default)
- [ ] Environment variables are configured (if needed)
- [ ] Resource requirements met (min 2GB RAM, 2 CPU cores recommended)

## Quick Deploy with Docker Compose (Recommended)

This is the fastest way to deploy the application:

```bash
# 1. Clone the repository
git clone <repository-url>
cd cityassist-ds

# 2. Build and start the service
docker-compose up --build -d

# 3. Verify deployment
curl http://localhost:8000/health

# 4. View logs
docker-compose logs -f

# 5. Stop the service
docker-compose down
```

## Manual Docker Deployment

If you prefer manual Docker commands:

```bash
# 1. Build the Docker image
docker build -t cityassist-ds-api:latest .

# 2. Run the container
docker run -d \
  --name cityassist-ds-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --restart unless-stopped \
  cityassist-ds-api:latest

# 3. Check status
docker ps
docker logs cityassist-ds-api

# 4. Stop and remove
docker stop cityassist-ds-api
docker rm cityassist-ds-api
```

## Kubernetes Deployment

For production-grade deployment with K8s:

### 1. Create ConfigMap (optional)

```yaml
# config-map.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cityassist-ds-config
data:
  LOG_LEVEL: "INFO"
```

```bash
kubectl apply -f config-map.yaml
```

### 2. Deploy the Application

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cityassist-ds-api
  labels:
    app: cityassist-ds-api
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
          name: http
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: cityassist-ds-config
              key: LOG_LEVEL
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
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
    protocol: TCP
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cityassist-ds-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cityassist-ds-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WORKERS` | `2` | Number of uvicorn workers |
| `PORT` | `8000` | API port |

## Resource Requirements

### Minimum
- CPU: 1 core
- RAM: 1GB
- Disk: 2GB (including models)

### Recommended
- CPU: 2 cores
- RAM: 2GB
- Disk: 5GB

### Production
- CPU: 4 cores
- RAM: 4GB
- Disk: 10GB
- Load Balancer
- Multiple replicas (3+)

## Health Checks

The API provides a `/health` endpoint for monitoring:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-11T10:00:00Z",
  "models": {
    "aqi": true,
    "traffic": true,
    "outage": true,
    "image": true
  }
}
```

## Monitoring Setup

### Prometheus Metrics

The API exposes metrics that can be scraped by Prometheus:

```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: 'cityassist-ds-api'
    static_configs:
      - targets: ['cityassist-ds-api-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics to Monitor

- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `model_inference_duration_seconds`: Model inference time
- `model_errors_total`: Model prediction errors

### Logging

Logs are structured in JSON format:

```bash
# View logs in Docker
docker logs -f cityassist-ds-api

# View logs in Kubernetes
kubectl logs -f deployment/cityassist-ds-api
```

## Testing Deployment

After deployment, run the test suite:

```bash
# If API is local
python scripts/test_api.py

# If API is remote
API_BASE_URL=http://your-api-url python scripts/test_api.py
```

## Troubleshooting

### Issue: Models not loading

**Symptoms**: API returns 500 errors or models show as `false` in health check

**Solution**:
```bash
# Train models first
cd cityassist-ds
python scripts/train_all_models.py

# Then redeploy
docker-compose up --build -d
```

### Issue: High memory usage

**Symptoms**: Container OOM kills or slow performance

**Solution**:
- Increase memory limits in Docker/K8s
- Reduce number of workers
- Enable model quantization

### Issue: Slow inference

**Symptoms**: High latency in predictions

**Solution**:
- Enable GPU support for image classification
- Implement caching layer (Redis)
- Scale horizontally (more replicas)

### Issue: Port already in use

**Symptoms**: Cannot bind to port 8000

**Solution**:
```bash
# Change port in docker-compose.yml
ports:
  - "8080:8000"  # Use 8080 instead

# Or stop conflicting service
docker ps
docker stop <container-id>
```

## Scaling Strategies

### Vertical Scaling
Increase resources per instance:
```yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Horizontal Scaling
Add more replicas:
```bash
# Docker Compose
docker-compose up --scale cityassist-ds-api=3

# Kubernetes
kubectl scale deployment cityassist-ds-api --replicas=5
```

### Load Balancing
Use nginx or K8s ingress:
```nginx
upstream cityassist_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://cityassist_api;
    }
}
```

## Security Considerations

1. **API Authentication**: Implement JWT or API key authentication
2. **HTTPS**: Always use HTTPS in production
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Input Validation**: All inputs are validated, but review for your use case
5. **Secrets Management**: Use environment variables or secret managers for sensitive data

## Backup and Recovery

### Backup Models
```bash
# Backup model artifacts
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to S3/GCS
aws s3 cp models-backup-*.tar.gz s3://your-bucket/backups/
```

### Restore Models
```bash
# Download from S3/GCS
aws s3 cp s3://your-bucket/backups/models-backup-20241111.tar.gz .

# Extract
tar -xzf models-backup-20241111.tar.gz
```

## Update Deployment

### Rolling Update (Kubernetes)
```bash
# Build new image
docker build -t cityassist-ds-api:v2 .

# Update deployment
kubectl set image deployment/cityassist-ds-api api=cityassist-ds-api:v2

# Monitor rollout
kubectl rollout status deployment/cityassist-ds-api

# Rollback if needed
kubectl rollout undo deployment/cityassist-ds-api
```

### Docker Compose Update
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up --build -d

# Zero-downtime update (manual)
docker-compose up -d --scale cityassist-ds-api=2
docker-compose stop cityassist-ds-api_1
docker-compose up -d --build cityassist-ds-api
docker-compose stop cityassist-ds-api_2
docker-compose scale cityassist-ds-api=1
```

## Performance Tuning

### Uvicorn Workers
```bash
# Increase workers for better throughput
uvicorn api.main:app --workers 4

# Or in Dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Model Optimization
- Use model quantization for smaller size
- Implement batch prediction for multiple requests
- Cache frequent predictions

## Support

For deployment issues:
1. Check logs: `docker logs` or `kubectl logs`
2. Verify health endpoint
3. Run test suite
4. Contact development team

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [API Documentation](http://localhost:8000/docs) (when running)

---

**Deployed successfully?** Run the test suite to verify: `python scripts/test_api.py`
