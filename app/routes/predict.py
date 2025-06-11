from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.core.model_loader import get_model
from app.utils.image_utils import preprocess_image
import numpy as np

router = APIRouter()

classes = [
    "Acne",
    "Blackhead", 
    "Blackhead Cystic Flat",
    "Blackhead Flat",
    "Conglobata",
    "Crystanlline"
    "Cystic",
    "Cystic Flat",
    "Flat",
    "Folliculitis",
    "Keloid",
    "Milium",
    "Papular",
    "Purulent",
    "Scars",
    "Sebo-crystan-conglo",
    "Syringoma",
    "Whitehead"
]

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = get_model()
        image_array = await preprocess_image(file)
        prediction = model.predict(image_array)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "class": classes[class_index],
            "confidence": confidence
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
