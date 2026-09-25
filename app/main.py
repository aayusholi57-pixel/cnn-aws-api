from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
from app.model import load_model
from app.preprocessing import preprocess_image

app = FastAPI(title="CNN MNIST API", description="Predicts handwritten digits")

# Load model once at startup to keep API fast
print("Loading CNN model...")
model = load_model()

@app.get("/")
def read_root():
    return {"message": "CNN API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # 1. Read the uploaded file
        image_bytes = await file.read()
        
        # 2. Convert to PyTorch tensor
        tensor = preprocess_image(image_bytes)
        
        # 3. Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        # 4. Return as JSON
        return {
            "prediction": predicted_class.item(),
            "confidence": round(confidence.item(), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))