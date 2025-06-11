from PIL import Image
import numpy as np
import io
from tensorflow.keras.preprocessing import image

IMG_SIZE = 64

async def preprocess_image(file):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array
