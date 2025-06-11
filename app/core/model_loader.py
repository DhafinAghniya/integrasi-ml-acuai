import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = 'model/model.h5'
GDRIVE_URL = 'https://drive.google.com/uc?id=1OOFw4ddZvo7rppsWL-AamAkUcSrc5zx_'

def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(">> Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print(">> Loading model.h5 ...")
    return load_model(MODEL_PATH)
