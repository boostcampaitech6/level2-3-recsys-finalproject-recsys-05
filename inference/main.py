from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger

from config import get_config, Config
from api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app")
    # Load model from model.py
    try:
        logger.info("Model loaded")
        # 모델 대신 matrix check
        yield
    except Exception as e:
        logger.exception("Error during startup")
        raise


app = FastAPI(lifespan=lifespan)
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
