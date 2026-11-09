
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import networkx as nx

from .config import RESULTS_DIR


SCHEMA_VERSION = 1

USABLE_STATUSES = {
    "red",
    "yellow",
    "clear",
    "closed",
}


def _empty_history() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "edges": {},
        "runs": [],
    }


def _edge_key(edge_id) -> str:
    return "|".join(map(str, edge_id))


def load_probe_history(path=None) -> dict:
    """Load previous probe outcomes, or create an empty history."""

    path = (
        Path(path)
        if path is not None
        else RESULTS_DIR / "probe_history.json"
    )

    if not path.exists():
        return _empty_history()

    history = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        not isinstance(history, dict)
        or history.get("schema_version") != SCHEMA_VERSION
        or not isinstance(history.get("edges"), dict)
        or not isinstance(history.get("runs"), list)
    ):
        raise ValueError("Invalid probe history format")

    return history


def update_probe_history(
    snapshot: nx.MultiDiGraph,
    summary: dict,
    history: dict | None = None,
) -> dict:
    """
    Record outcomes for directly queried edges only.

    Reverse-inferred observations are recorded separately.
    Global API failures do not penalize a particular road candidate.
    """

    if not isinstance(snapshot, nx.MultiDiGraph):
        raise TypeError("snapshot must be a MultiDiGraph")

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary")

    history = _empty_history() if history is None else history

    if not isinstance(history.get("edges"), dict):
        raise ValueError("Invalid history edges")

    requests_made = int(summary.get("requests_made", 0))

    # Do not record offline runs as real collection attempts.
    if requests_made <= 0:
        return history

    now = datetime.now(timezone.utc).isoformat()

    run_direct = 0
    run_usable = 0
    run_reverse = 0

    for u, v, key, attrs in snapshot.edges(
        keys=True,
        data=True,
    ):

        if not attrs.get("traffic_directly_queried"):
            continue

        edge_id = (u, v, key)
        key_text = _edge_key(edge_id)

        item = history["edges"].setdefault(
            key_text,
            {
                "attempts": 0,
                "direct_usable": 0,
                "direct_matched": 0,
                "reverse_rescued": 0,
                "global_errors": 0,
                "point_errors": 0,
                "geometry_failures": 0,
                "quality_failures": 0,
            },
        )

        item["attempts"] += 1
        run_direct += 1

        reason = str(attrs.get("traffic_reason") or "")
        match_reason = str(
            attrs.get("traffic_match_reason") or ""
        )

        status = attrs.get("traffic_status")
        source = attrs.get("traffic_match_source")

        if (
            source == "direct_probe"
            and status in USABLE_STATUSES
        ):
            item["direct_usable"] += 1
            run_usable += 1

        if (
            source == "direct_probe"
            and attrs.get("traffic_matched")
        ):
            item["direct_matched"] += 1

        if reason == "matched_on_reverse_edge":
            item["reverse_rescued"] += 1
            run_reverse += 1

        if reason == "api_network_error":
            item["global_errors"] += 1

        elif reason.startswith("api_http_"):
            try:
                code = int(reason.rsplit("_", 1)[-1])
            except ValueError:
                code = None

            if code in (401, 403, 429):
                item["global_errors"] += 1
            elif code is not None and code >= 500:
                item["global_errors"] += 1
            else:
                item["point_errors"] += 1

        elif reason in {
            "geometry_or_heading_mismatch",
            "heading_proxy_mismatch",
            "midpoint_too_far",
            "probe_point_too_far",
            "insufficient_overlap",
            "insufficient_local_geometry",
            "short_segment",
            "invalid_tomtom_geometry",
            "geometry_transform_error",
            "osm_edge_not_found",
            "ambiguous_reverse_match",
        }:
            item["geometry_failures"] += 1

        elif reason in {
            "low_confidence",
            "missing_confidence",
            "invalid_or_zero_speed",
        }:
            item["quality_failures"] += 1

        item["last_reason"] = reason
        item["last_match_reason"] = match_reason
        item["last_status"] = status
        item["last_attempt_utc"] = now

    history["runs"].append({
        "timestamp_utc": now,
        "requests_made": requests_made,
        "request_errors": int(
            summary.get("request_errors", 0)
        ),
        "direct_queries": run_direct,
        "direct_usable": run_usable,
        "reverse_rescued": run_reverse,
    })

    # Keep the history file bounded.
    history["runs"] = history["runs"][-200:]

    return history


def save_probe_history(history: dict, path=None) -> Path:
    """Save history atomically."""

    path = (
        Path(path)
        if path is not None
        else RESULTS_DIR / "probe_history.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            history,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)

    return path
