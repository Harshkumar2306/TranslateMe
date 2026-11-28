"""
TranslateMe FastAPI Application Entrypoint
Universal Multilingual Neural Machine Translation System
"""

import os
import sys
import socket
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings, REPO_ROOT
from backend.app.database.session import Base, engine
from backend.app.api.routes import router
from backend.app.services.custom_multilingual_service import custom_multilingual_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("translateme")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: Initializes SQLite and loads our custom-trained
    Multilingual Transformer model from scratch.
    """
    logger.info("Initializing TranslateMe Custom Transformer Deep Learning Services...")
    try:
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
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def cache_control_middleware(request: Request, call_next):
    """Prevents stale browser caching of dynamic API payloads and SPA index.html."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith(settings.API_PREFIX):
        response.headers["Cache-Control"] = "no-store"
    elif path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# Include API Router
app.include_router(router, prefix=settings.API_PREFIX)

# Serve compiled Frontend static assets resolved from repository root
frontend_dist_path = str(REPO_ROOT / "frontend" / "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"[Error ID: {error_id}] Global unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error_id": error_id},
    )


def _is_port_in_use(port: int) -> bool:
    """Checks 127.0.0.1 and localhost so macOS specific-IP listeners cannot silently shadow 0.0.0.0."""
    for addr in ("127.0.0.1", "::1"):
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((addr, port)) == 0:
                    return True
        except OSError:
            continue
    return False


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    if _is_port_in_use(port):
        logger.error(
            f"Port {port} is already in use (e.g. an older server on 127.0.0.1:{port}). "
            f"Stop it first (lsof -i :{port}) before starting TranslateMe."
        )
        sys.exit(1)

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
