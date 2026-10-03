"""Vercel entrypoint serving the Prog3 FastAPI application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Prog3"))

from backend_api import app, init_database

# Schema and seed data are ensured here as well as in the lifespan so they
# exist even when the host skips ASGI lifespan startup.
init_database()
