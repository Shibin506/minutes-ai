"""Vercel entry point: the FastAPI app as one Python function behind /api/*.

Serverless defaults: a throwaway SQLite database in /tmp, the sample meetings loaded on a cold
start (with their saved analyses, so no model call), and uploads analyzed inside the request
(there is no background worker once the response has been sent).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("MINUTES_DB_PATH", "/tmp/minutes.db")
os.environ.setdefault("MINUTES_SEED_SAMPLES", "1")
os.environ.setdefault("MINUTES_SYNC_ANALYSIS", "1")
os.environ.setdefault("MINUTES_SAMPLES_DIR", str(ROOT / "samples"))
os.environ.setdefault("MINUTES_RETRIEVAL_K", "6")

from minutes.api import create_app  # noqa: E402

app = create_app()
