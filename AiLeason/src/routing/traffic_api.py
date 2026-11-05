
from __future__ import annotations

from math import isfinite
import re
import requests

from .config import tomtom_api_key

FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/18/json"


def _sanitize(value: object, api_key: str) -> str:
    message = str(value)[:500]
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"(?i)(?:key|api[_-]?key|token)\s*[=:]\s*[^\s&,;]+", "credential=[REDACTED]", message)
    message = re.sub(r"https?://[^\s]+", "[URL REDACTED]", message)
    return message[:180]


class TomTomAPIError(RuntimeError):
    def __init__(
        self,
        status_code: int | None = None,
        *,
        error_code: str | None = None,
        detail: str | None = None,
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        status = "network failure" if status_code is None else f"HTTP {status_code}"
        extra = f"; {error_code}" if error_code else ""
        super().__init__(f"TomTom Flow Segment failed ({status}{extra}; credentials hidden)")


class TomTomClient:
    def __init__(self, api_key: str | None = None, *, timeout_s: float = 15.0, zoom: int = 18):
        self._key = api_key if api_key is not None else tomtom_api_key()
        if not isinstance(self._key, str) or not self._key.strip():
            raise ValueError("Missing TomTom API key")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not 0 <= zoom <= 22:
            raise ValueError("zoom must be in 0..22")
        self.timeout_s = timeout_s
        self.zoom = zoom
        self.session = requests.Session()

    def fetch_flow(self, lat: float, lon: float) -> dict:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError) as exc:
            raise ValueError("Coordinates must be numeric") from exc
        if not (isfinite(lat) and isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Coordinates must be finite WGS84 latitude and longitude")
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{self.zoom}/json"
        try:
            response = self.session.get(
                url,
                params={"key": self._key, "point": f"{lat:.7f},{lon:.7f}", "unit": "kmph", "openLr": "false"},
                timeout=self.timeout_s,
            )
        except requests.RequestException:
            raise TomTomAPIError() from None
        if not response.ok:
            code = detail = None
            try:
                body = response.json()
                d = body.get("detailedError", {}) if isinstance(body, dict) else {}
                if isinstance(d, dict):
                    code = _sanitize(d.get("code", ""), self._key) or None
                    detail = _sanitize(d.get("message", ""), self._key) or None
            except (ValueError, TypeError):
                pass
            raise TomTomAPIError(response.status_code, error_code=code, detail=detail)
        try:
            payload = response.json()["flowSegmentData"]
        except (ValueError, KeyError, TypeError):
            raise TomTomAPIError(response.status_code, error_code="INVALID_RESPONSE") from None
        if not isinstance(payload, dict):
            raise TomTomAPIError(response.status_code, error_code="INVALID_RESPONSE")
        return payload

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "TomTomClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
