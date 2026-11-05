
from pathlib import Path
from dotenv import load_dotenv
import os

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



MAX_TRAFFIC_REQUESTS = 60
MIN_TOMTOM_CONFIDENCE = 0.65
DEFAULT_UNOBSERVED_SPEED_KPH = 30.0 
TRAFFIC_REQUEST_PAUSE_S = 0.1
MATCH_MIDPOINT_TOLERANCE_M = 12.0
MATCH_SAMPLE_TOLERANCE_M = 15.0
MATCH_MIN_COSINE = 0.80 


def tomtom_api_key() -> str:
    key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TOMTOM_API_KEY is missing. Put it in project-root .env")
    return key
