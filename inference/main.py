from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger

from config import config
from load_model import load_model
from api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app")
    # Load model from model.py
    logger.info("Loading model")
    load_model(config.model_path)
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
