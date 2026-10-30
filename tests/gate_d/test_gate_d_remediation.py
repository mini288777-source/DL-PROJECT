"""
Tests for Gate D Remediation: Runtime Boundaries, Cybersecurity Controls, and Fail-Safe Safeguards.
"""

import numpy as np
import pytest

from src.app.config import (
    APP_HOST,
    CORS_ORIGINS,
    MAX_UPLOAD_BYTES,
    MAX_AUDIO_DURATION_SEC,
    DISCLAIMER_NOTICE,
    SCIENTIFIC_VERDICT_NOTICE,
    DEFAULT_DECISION_TAU,
    FER_SAFETY_BOUND
)
from src.app.model_manager import ModelManager, ModelNotReadyException
from src.app.routes import (
    health_router,
    audio_router,
    inference_router,
    websocket_router,
    research_router
)


def test_loopback_and_network_binding_controls():
    """Verify application binds strictly to loopback 127.0.0.1 by default."""
    assert APP_HOST in ("127.0.0.1", "localhost")
    # Verify CORS origins are strictly loopback / local development
    for origin in CORS_ORIGINS:
        assert "127.0.0.1" in origin or "localhost" in origin


def test_upload_and_resource_limits():
    """Verify upload size and duration controls prevent resource exhaustion."""
    assert MAX_UPLOAD_BYTES <= 10 * 1024 * 1024  # 10 MB limit
    assert MAX_AUDIO_DURATION_SEC <= 30.0  # 30 seconds max duration


def test_unmaterialized_checkpoint_raises_503_exception():
    """Verify unmaterialized Phase 8 candidate models raise ModelNotReadyException (HTTP 503)."""
    model_mgr = ModelManager.get_instance()
    unready_id = "phase8_1_track_a_baseline"
    
    # Must report unmaterialized in model info
    info = model_mgr.get_model_info(unready_id, "F01")
    assert info["status"] == "MODEL_NOT_READY"
    assert info["checkpoint_provenance"]["exists"] is False
    
    # Predict call must raise ModelNotReadyException (which maps to HTTP 503)
    with pytest.raises(ModelNotReadyException) as exc_info:
        model_mgr.predict(np.zeros(384, dtype=np.float32), speaker_id="F01", model_id=unready_id)
    assert exc_info.value.model_id == unready_id
    assert "MODEL_NOT_READY" in exc_info.value.message or "unmaterialized" in exc_info.value.message


def test_no_external_dispatch_or_telephony_routes():
    """Verify zero external dispatch, 911, telephony, or alert endpoints exist in the API."""
    all_routers = [health_router, audio_router, inference_router, websocket_router, research_router]
    routes = [getattr(r, "path", "") for router in all_routers for r in router.routes]
    prohibited_keywords = ["dispatch", "911", "call_center", "sms", "alert_external", "ems", "twilio"]
    for route_path in routes:
        for kw in prohibited_keywords:
            assert kw not in route_path.lower(), f"Prohibited endpoint found in API routes: {route_path}"


def test_disclaimers_and_scientific_verdict_notices():
    """Verify mandatory disclaimers and Case B scientific verdicts are defined."""
    assert "experimental research prototype" in DISCLAIMER_NOTICE.lower()
    assert "not validated for clinical use" in DISCLAIMER_NOTICE.lower()
    assert "case b affirmed" in SCIENTIFIC_VERDICT_NOTICE.lower()
    assert "no statistically supported quantum advantage" in SCIENTIFIC_VERDICT_NOTICE.lower()


def test_threshold_tau_decoupled_from_internal_fer_bound():
    """Verify tau is decoupled from internal FER bound and FER bound is 0.1500."""
    assert DEFAULT_DECISION_TAU == 0.6000
    assert FER_SAFETY_BOUND == 0.1500
    assert DEFAULT_DECISION_TAU != FER_SAFETY_BOUND
