"""
FastAPI application entry point.
Production-grade setup with CORS, exception handlers, and health check.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from backend.routers.image_router import router as image_router
from backend.routers.video_router import router as video_router
from backend.schemas.response_schema import HealthResponse, ErrorResponse
from utils.config import settings
from utils.logger import logger

# ──────────────────────────────────────────────────────────────────────────────
# App initialization
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ──────────────────────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, and response time."""
    t_start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({elapsed_ms}ms)"
    )
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
        ).model_dump(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

app.include_router(image_router)
app.include_router(video_router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check endpoint."""
    return HealthResponse()


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Color Detection & Tracking API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
