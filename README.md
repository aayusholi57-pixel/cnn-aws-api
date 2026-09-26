# CNN Image Classification API

A production-style PyTorch CNN inference API built with **FastAPI, Docker, GitHub Actions, and AWS EC2**. The project is API-only; Swagger/OpenAPI is the intended interface.

## Architecture

```
Client
  ↓
FastAPI
  ↓
Image validation + preprocessing
  ↓
PyTorch CNN
  ↓
Prediction + confidence
```

Deployment:

```
GitHub
  ↓
GitHub Actions
  ├── pytest
  ├── Docker build
  └── AWS EC2 deployment
        ↓
      Docker
        ↓
     FastAPI :9000
        ↓
     EC2 :8080
```

The deployment workflow builds the Linux image on GitHub Actions and transfers the finished image to EC2. EC2 does not build PyTorch locally.

## API

### Health

```http
GET /health
GET /api/v1/health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Swagger / OpenAPI

Open `http://127.0.0.1:9000/docs` locally or `http://<EC2_PUBLIC_IP>:8080/docs` after deployment. Swagger lets you upload an image and call the prediction endpoint directly, so no separate frontend is required.

### Prediction

```http
POST /api/v1/predict
```

The legacy `/predict` route is also retained for compatibility.

Upload an image as multipart form data using the field name `file`.

Example:

```bash
curl -X POST "http://<EC2_PUBLIC_IP>:8080/api/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.png"
```

Response:

```json
{
  "filename": "sample.png",
  "prediction": 3,
  "confidence": 0.982341
}
```

The model currently exposes 10 numeric output classes. Class names are intentionally not invented because the training-label mapping is not stored in this repository.

## Local development

Use Python 3.10.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

Then open:

- `http://127.0.0.1:9000/`
- `http://127.0.0.1:9000/health`
- `http://127.0.0.1:9000/docs`

## Docker

Build:

```bash
docker build -t cnn-api:latest .
```

Run:

```bash
docker run --rm -p 8080:9000 cnn-api:latest
```

The image includes a Docker health check against `/health` and runs the application as a non-root user.

## CI/CD

Every pull request and push to `main` runs:

1. Python dependency installation
2. Pytest API/inference tests
3. Docker image build

Only a successful push to `main` proceeds to the EC2 deployment job. GitHub Actions deployment concurrency prevents overlapping deployments.

Required GitHub Actions secrets:

- `EC2_HOST`
- `EC2_USERNAME`
- `EC2_SSH_KEY`

Configure production deployment secrets in the GitHub **production environment**.

## AWS EC2

Allow inbound TCP `8080` in the EC2 security group if the API should be publicly reachable.

The deployment uses a candidate container on port `9001`, verifies its health endpoint, then promotes it to port `8080`. If the promoted container fails its health check, the previous container is restored.

For production internet traffic, place the EC2 service behind HTTPS/TLS and a reverse proxy or load balancer rather than exposing Uvicorn directly.

## Security and reliability

- Uploaded images are validated with Pillow before inference.
- Uploads are limited to 10 MB.
- PyTorch model loading uses `weights_only=True`.
- The container runs as a non-root user.
- Docker uses an explicit health check.
- CI runs before deployment.
- Deployment performs candidate health checks and rollback.
- No credentials are stored in the repository.

## Project structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── preprocessing.py
├── models/
│   └── cnn_model.pth
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── deploy.yml
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Important model note

The repository contains the trained weights and inference architecture, but not the original training dataset or label metadata. The API therefore returns the model's numeric class index instead of claiming human-readable labels that cannot be verified from the repository.

## Next production steps

For a larger deployment, move container publishing to a registry such as Amazon ECR and place the EC2 service behind an HTTPS load balancer. The current project intentionally keeps the deployment path simple enough to understand and operate while still providing automated testing and rollback.
