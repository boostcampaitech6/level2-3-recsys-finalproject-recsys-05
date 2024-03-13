from fastapi import FastAPI
from contextlib import contextmanager
from loguru import logger

from config import config
from load_model import load_model

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

async def lifespan(app: FastAPI) -> None:
    logger.info()
    # Load model from model.py
    logger.info("Loading model")
    load_model(config.model_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
