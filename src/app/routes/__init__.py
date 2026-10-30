"""API route modules for Gate B Prototype."""

from src.app.routes.health import router as health_router
from src.app.routes.audio import router as audio_router
from src.app.routes.inference import router as inference_router
from src.app.routes.websocket import router as websocket_router
from src.app.routes.research import router as research_router

__all__ = [
    "health_router",
    "audio_router",
    "inference_router",
    "websocket_router",
    "research_router",
]
