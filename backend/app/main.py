"""
TranslateMe FastAPI Application Entrypoint
Universal Multilingual Neural Machine Translation System
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.database.session import Base, engine
from backend.app.api.routes import router
from backend.app.services.custom_multilingual_service import custom_multilingual_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("translateme")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: Loads our custom-trained Multilingual Transformer model
    from scratch without downloading or depending on external pretrained models.
    """
    logger.info("Initializing TranslateMe Custom Transformer Deep Learning Services...")
    try:
        # Initialize SQLite database schema safely during app lifespan
        Base.metadata.create_all(bind=engine)
        custom_multilingual_engine.load_model()
        logger.info("TranslateMe Custom Transformer startup complete. Ready to serve requests!")
    except Exception as e:
        logger.warning(f"Custom model loading notification: {e}")
    yield
    logger.info("Shutting down TranslateMe Custom Translation Services.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Custom Multilingual Neural Machine Translation System Built and Trained From Scratch in PyTorch.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(router, prefix=settings.API_PREFIX)

# Serve compiled Frontend static assets if available
frontend_dist_path = os.path.abspath("frontend/dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")


import uuid

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"[Error ID: {error_id}] Global unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error_id": error_id}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
