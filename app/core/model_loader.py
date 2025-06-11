from tensorflow.keras.models import load_model

_model = None

def get_model():
    global _model
    if _model is None:
        print(">> Loading model.h5 ...")
        _model = load_model("model/model.h5")
        print(">> Model loaded successfully.")
    return _model
