from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.app.audio_service import AudioValidationError, AudioProcessingError
from src.app.config import (
    APP_HOST,
    APP_PORT,
    CORS_ORIGINS,
    DISCLAIMER_NOTICE,
    REPO_ROOT,
)
from src.app.model_manager import ModelNotReadyException, ModelNotFoundError
from src.app.quantum_model_manager import (
    QuantumModelNotReadyException,
    ModelNotAllowedException,
    ModelNotFoundError as QuantumModelNotFoundError,
)
from src.app.routes import (
    health_router,
    audio_router,
    inference_router,
    websocket_router,
    research_router,
)
from src.app.schemas import current_utc_timestamp

# Configure root app logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gate_b_app")

STATIC_DIR = REPO_ROOT / "src" / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Gate B.2 Research Demonstration Application...")
    logger.info("Verifying checkpoint inventory, Whisper feature extraction, and research routes...")
    logger.info("Research Disclaimer: For controlled research demonstration only. Not for clinical or dispatch use.")
    yield
    logger.info("Shutting down Gate B.2 Research Demonstration Application.")

app = FastAPI(
    title="Atypical Speech Emergency Assistant — Gate B.2 Research Demonstration",
    description=(
        "Full-featured local research demonstration application demonstrating personalized speech processing, "
        "Whisper feature extraction, offline ASR, and continual classical adapter inference "
        "for controlled research demonstration only. "
        "Strictly decoupled decision cutoff tau from the 15% FER safety ceiling."
    ),
    version="1.1.0-gate-b2",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration restricted to local development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(ModelNotReadyException)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyException):
    """Returns HTTP 503 Service Unavailable when unmaterialized weights are requested."""
    logger.warning(f"503 Service Unavailable: {exc.model_id} requested but checkpoint is unmaterialized.")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error_code": "MODEL_NOT_READY",
            "model_id": exc.model_id,
            "message": exc.message,
            "remediation": exc.remediation,
            "disclaimer": DISCLAIMER_NOTICE,
            "timestamp_utc": current_utc_timestamp(),
        },
    )


@app.exception_handler(ModelNotAllowedException)
async def model_not_allowed_handler(request: Request, exc: ModelNotAllowedException):
    """Returns HTTP 400 Bad Request when classical models are requested for operational inference."""
    logger.warning(f"400 Model Not Allowed: {exc.model_id} requested for operational prediction.")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": exc.error_code,
            "model_id": exc.model_id,
            "message": exc.message,
            "disclaimer": DISCLAIMER_NOTICE,
            "timestamp_utc": current_utc_timestamp(),
        },
    )


@app.exception_handler(QuantumModelNotReadyException)
async def quantum_model_not_ready_handler(request: Request, exc: QuantumModelNotReadyException):
    """Returns HTTP 503 Service Unavailable when quantum model state is unmaterialized or incomplete."""
    logger.warning(f"503 Quantum Model Not Ready: {exc.model_id} - {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error_code": exc.error_code,
            "model_id": exc.model_id,
            "message": exc.message,
            "remediation": exc.remediation,
            "artifact_path": exc.artifact_path,
            "disclaimer": DISCLAIMER_NOTICE,
            "timestamp_utc": current_utc_timestamp(),
        },
    )


@app.exception_handler(AudioValidationError)
async def audio_validation_handler(request: Request, exc: AudioValidationError):
    """Returns HTTP 400 Bad Request on invalid audio input without exposing sensitive traces."""
    logger.warning(f"400 Audio Validation Error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "AUDIO_VALIDATION_ERROR",
            "message": str(exc),
            "disclaimer": DISCLAIMER_NOTICE,
            "timestamp_utc": current_utc_timestamp(),
        },
    )


@app.exception_handler(ModelNotFoundError)
@app.exception_handler(QuantumModelNotFoundError)
async def model_not_found_handler(request: Request, exc: Exception):
    """Returns HTTP 404 Not Found for unknown model or speaker identifiers."""
    error_code = getattr(exc, "error_code", "MODEL_NOT_FOUND")
    logger.warning(f"404 Not Found: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error_code": error_code,
            "message": str(exc),
            "timestamp_utc": current_utc_timestamp(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches unhandled runtime exceptions without exposing internal stack traces."""
    logger.error(f"500 Internal Server Error: {str(exc)}", exc_info=False)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected server error occurred during request processing.",
            "disclaimer": DISCLAIMER_NOTICE,
            "timestamp_utc": current_utc_timestamp(),
        },
    )


# Include Route Modules
app.include_router(health_router)
app.include_router(audio_router)
app.include_router(inference_router)
app.include_router(websocket_router)
app.include_router(research_router)

# Mount Static Assets & Verified Research Figures
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

readiness_fig_dir = REPO_ROOT / "reports" / "project_readiness" / "figures"
if readiness_fig_dir.exists():
    app.mount("/figures/readiness", StaticFiles(directory=readiness_fig_dir), name="readiness_figures")

phase8_fig_dir = REPO_ROOT / "results" / "phase8_1" / "figures"
if phase8_fig_dir.exists():
    app.mount("/figures/phase8_1", StaticFiles(directory=phase8_fig_dir), name="phase8_1_figures")

samples_dir = REPO_ROOT / "data" / "synthetic_emergency_audio"
if samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=samples_dir), name="synthetic_samples")


@app.get("/", include_in_schema=False)
@app.get("/demo", include_in_schema=False)
async def serve_demo_interface():
    """Serves the controlled research demonstration web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "running", "docs": "/docs"})
