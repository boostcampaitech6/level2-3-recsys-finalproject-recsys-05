from loguru import logger

model = None

def load_model(model_path: str):
    import joblib

    logger.info(f"Loading model from {model_path}")

    global model
    model = joblib.load(model_path)

def get_model():
    global model
    return model
