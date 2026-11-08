from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# This file is expected at: AiLeason/src/routing/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
GRAPH_PATH = RAW_DIR / "south_kensington_drive.graphml"

MAP_CENTER = (51.4994, -0.1760)
MAP_RADIUS_M = 1200

ORIGIN = (51.4946, -0.1742)
ORIGINS = {
    "South Kensington": (51.4946, -0.1742),
    "Royal Albert Hall": (51.5009, -0.1777),
    "Gloucester Road": (51.4953, -0.1825),
}
DESTINATION = (51.4990, -0.1742)

# TomTom collection controls.
MAX_TRAFFIC_REQUESTS = 60
MIN_TOMTOM_CONFIDENCE = 0.65
DEFAULT_UNOBSERVED_SPEED_KPH = 30.0
TRAFFIC_REQUEST_PAUSE_S = 0.5

# Probe planning. Very short OSM edges are not reliable Flow Segment probes.
MIN_PROBE_EDGE_LENGTH_M = 35.0
MAX_PROBE_ATTEMPTS_PER_EDGE = 3
PROBE_FRACTIONS = (0.50, 0.25, 0.75)

# Map-matching controls. Distances are measured in EPSG:27700 metres.
# Keep these conservative: increasing them too far can match a parallel road.
MATCH_MIDPOINT_TOLERANCE_M = 12.0
MATCH_SAMPLE_TOLERANCE_M = 15.0
MATCH_MIN_COSINE = 0.80
MATCH_LOCAL_WINDOW_M = 25.0
MATCH_MIN_LOCAL_COVERAGE = 0.80
MIN_TOMTOM_SEGMENT_LENGTH_M = 5.0


def tomtom_api_key() -> str:
    """Return the TomTom API key from the project-root .env file."""
    key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "TOMTOM_API_KEY is missing. Put it in the AiLeason project-root .env"
        )
    return key
