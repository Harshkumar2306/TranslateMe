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
from backend.app.api.routes import router
from backend.app.services.nllb_service import translation_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("translateme")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: warms up the multilingual Transformer model once
    upon server start so users experience fast translation without cold-start delay.
    """
    logger.info("Initializing TranslateMe Deep Learning Translation Services...")
    try:
        translation_engine.load_model()
        logger.info("TranslateMe Backend startup complete. Ready to serve requests!")
    except Exception as e:
        logger.warning(f"Deferred model loading: {e}")
    yield
    logger.info("Shutting down TranslateMe Translation Services.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Universal Multilingual Neural Machine Translation System Using Transformer-Based Deep Learning (NLLB-200 & Custom Transformer).",
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
