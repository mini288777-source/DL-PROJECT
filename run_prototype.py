"""Local runner script for Gate B Engineering Prototype.

Usage:
    python run_prototype.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

import argparse
import sys
from pathlib import Path
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.app.config import APP_HOST, APP_PORT, DISCLAIMER_NOTICE


def main():
    parser = argparse.ArgumentParser(
        description="Run the Gate B Engineering Prototype for Controlled Research Demonstration Only."
    )
    parser.add_argument("--host", default=APP_HOST, help=f"Host address (default: {APP_HOST})")
    parser.add_argument("--port", type=int, default=APP_PORT, help=f"Port (default: {APP_PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    args = parser.parse_args()

    print("=" * 80)
    print(" ATYPICAL SPEECH EMERGENCY ASSISTANT — GATE B ENGINEERING PROTOTYPE")
    print(" Controlled Research Demonstration Only")
    print("=" * 80)
    print(f"Server Host: http://{args.host}:{args.port}")
    print(f"Interactive Web Demo: http://{args.host}:{args.port}/demo")
    print(f"Interactive API Docs: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print(f"Readiness Check: http://{args.host}:{args.port}/ready")
    print("-" * 80)
    print(f"DISCLAIMER: {DISCLAIMER_NOTICE}")
    print("=" * 80)

    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
