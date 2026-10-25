

from __future__ import annotations

from math import isfinite

import requests

from .config import tomtom_api_key

FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/18/json"


class TomTomAPIError(RuntimeError):
    def __init__(self, status_code: int | None = None):
        self.status_code = status_code
        message = "network failure" if status_code is None else f"HTTP {status_code}"
        super().__init__(f"TomTom Flow Segment request failed ({message}; credentials hidden)")


class TomTomClient:


    def __init__(self, api_key: str | None = None, *, timeout_s: float = 15.0):
        self._key = api_key if api_key is not None else tomtom_api_key()
        if not self._key or not self._key.strip():
            raise ValueError("Missing TomTom API key")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def fetch_flow(self, lat: float, lon: float) -> dict:
        if not all(isfinite(float(x)) for x in (lat, lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Coordinates must be finite WGS84 latitude and longitude")
        try:
            response = self.session.get(
                FLOW_URL,
                params={
                    "key": self._key,
                    "point": f"{lat:.7f},{lon:.7f}", 
                    "unit": "kmph",
                    "openLr": "false", 
                },
                timeout=self.timeout_s,
            )
        except requests.RequestException:

            raise TomTomAPIError() from None
        if not response.ok:
            raise TomTomAPIError(response.status_code)
        try:
            item = response.json()["flowSegmentData"]
        except (ValueError, KeyError, TypeError):
            raise TomTomAPIError(response.status_code) from None
        if not isinstance(item, dict):
            raise TomTomAPIError(response.status_code)
        return item

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "TomTomClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
