from pathlib import Path
import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model import load_model
from app.preprocessing import preprocess_image

APP_VERSION = "1.0.0"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="CNN Image Classification API",
    version=APP_VERSION,
    description="Production-ready FastAPI service for CNN image classification.",
)


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_model.pth"
model = load_model(str(MODEL_PATH))


@app.get("/", tags=["system"])
def home():
    return {
        "message": "CNN API is running successfully!",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["system"])
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": APP_VERSION,
    }


@app.get("/api/v1/health", tags=["system"])
def api_health():
    return health()


@app.post("/predict", tags=["inference"])
@app.post("/api/v1/predict", tags=["inference"])
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()

        tensor = preprocess_image(contents)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    with torch.inference_mode():
        probabilities = F.softmax(model(tensor), dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)

    return {
        "filename": file.filename,
        "prediction": predicted_class.item(),
        "confidence": round(confidence.item(), 6),
    }
