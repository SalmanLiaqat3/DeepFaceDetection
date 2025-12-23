# Docker Deployment Guide

This guide explains how to build and run the Face Recognition application using Docker.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose (optional, for easier deployment)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the application:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t face-recognition-app:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name face-recognition-app \
     -p 5000:5000 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/static/results:/app/static/results \
     face-recognition-app:latest
   ```

3. **View logs:**
   ```bash
   docker logs -f face-recognition-app
   ```

4. **Stop the container:**
   ```bash
   docker stop face-recognition-app
   docker rm face-recognition-app
   ```

## Accessing the Application

Once the container is running, access the application at:
- **URL:** http://localhost:5000

## Volumes

The Docker setup uses volumes to persist data:
- `./data` - Stores user images, embeddings, and attendance records
- `./static/results` - Stores detection result images

## Production Considerations

### Environment Variables

You can customize the deployment using environment variables:

```bash
docker run -d \
  -e FLASK_ENV=production \
  -e GUNICORN_WORKERS=4 \
  -p 5000:5000 \
  face-recognition-app:latest
```

### Resource Limits

The docker-compose.yml includes resource limits. Adjust them based on your server capacity:
- **CPU:** 1-2 cores recommended
- **Memory:** 2-4GB recommended

### Scaling

To run multiple instances behind a load balancer:

```bash
docker-compose up -d --scale face-recognition-app=3
```

Note: Ensure your load balancer handles sticky sessions if needed.

### Health Checks

The container includes a health check that verifies the application is responding. Check health status:

```bash
docker ps  # Look for "healthy" status
```

## Troubleshooting

### Container won't start

1. Check logs:
   ```bash
   docker logs face-recognition-app
   ```

2. Verify model files exist:
   ```bash
   ls -la model/
   ```

3. Check file permissions:
   ```bash
   docker exec face-recognition-app ls -la /app
   ```

### Performance Issues

1. Increase worker count in Dockerfile CMD:
   ```dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "8", ...]
   ```

2. Adjust resource limits in docker-compose.yml

3. Use a reverse proxy (nginx) for static file serving

### Model Loading Errors

Ensure all model files are present:
- `model/facetracker_balanced.h5`
- `model/facenet_keras (1).h5`

## Building for Different Platforms

### Build for Linux/AMD64:
```bash
docker build --platform linux/amd64 -t face-recognition-app:latest .
```

### Build for ARM64 (Apple Silicon, Raspberry Pi):
```bash
docker build --platform linux/arm64 -t face-recognition-app:latest .
```

## Security Notes

- The container runs as a non-root user (`appuser`)
- Only necessary ports are exposed
- System dependencies are minimized
- Health checks are configured

## Updating the Application

1. **Pull latest code:**
   ```bash
   git pull
   ```

2. **Rebuild the image:**
   ```bash
   docker-compose build
   ```

3. **Restart the service:**
   ```bash
   docker-compose up -d
   ```

