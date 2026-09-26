from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "CNN API is running successfully!"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_versioned_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_cors_for_browser_frontend():
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_invalid_image():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("broken.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_prediction():
    image = Image.new("L", (28, 28), color=0)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in range(10)
    assert 0.0 <= body["confidence"] <= 1.0
