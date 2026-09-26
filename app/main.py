from pathlib import Path
import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model import load_model
from app.preprocessing import preprocess_image

app = FastAPI(title="CNN Image Classification API")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_model.pth"
model = load_model(str(MODEL_PATH))

@app.get("/")
def home():
    return {"message": "CNN API is running successfully!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        # Validate the upload before passing the original bytes through preprocessing.
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
        tensor = preprocess_image(contents)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    with torch.inference_mode():
        probabilities = F.softmax(model(tensor), dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)

    return {
        "filename": file.filename,
        "prediction": predicted_class.item(),
        "confidence": confidence.item(),
    }
