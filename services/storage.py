"""
Redis intermediate storage between pipeline workers.
All keys are scoped to job_id and auto-expire.
Deleted entirely after report is stored to FileMaker.
"""
import json
import pickle
import zlib
from typing import Any
import numpy as np
from redis import Redis
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

_DATA      = "job:{j}:data"
_RESPONSES = "job:{j}:responses"
_VECTORS   = "job:{j}:vectors"
_CLUSTERS  = "job:{j}:clusters"
_ANALYSIS  = "job:{j}:analysis"
_STATE     = "job:{j}:state"
_METRICS   = "job:{j}:metrics"
_STAGES    = "job:{j}:stages"


def _ttl() -> int:
    return get_settings().redis.result_ttl


def _compress_json(data: Any) -> bytes:
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return zlib.compress(raw, level=6)


def _decompress_json(raw: bytes) -> Any:
    try:
        return json.loads(zlib.decompress(raw).decode("utf-8"))
    except Exception:
        # Backward compatibility for older uncompressed payloads
        return json.loads(raw)


def _compress_pickle(data: Any) -> bytes:
    return zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=6)


def _decompress_pickle(raw: bytes) -> Any:
    try:
        return pickle.loads(zlib.decompress(raw))
    except Exception:
        return pickle.loads(raw)


# ── Full FM data dict ────────────────────────────────────────────────

def store_data(conn: Redis, job_id: str, data: dict[str, Any]) -> None:
    conn.setex(_DATA.format(j=job_id), _ttl(), _compress_json(data))
    logger.info("stored_data", job_id=job_id)


def load_data(conn: Redis, job_id: str) -> dict[str, Any]:
    raw = conn.get(_DATA.format(j=job_id))
    if not raw:
        raise RuntimeError(f"No FM data in Redis for job {job_id}")
    return _decompress_json(raw)


# ── Responses ────────────────────────────────────────────────────────

def store_responses(conn: Redis, job_id: str, data: dict[str, Any]) -> None:
    conn.setex(_RESPONSES.format(j=job_id), _ttl(), _compress_json(data))
    logger.info("stored_responses", job_id=job_id)


def load_responses(conn: Redis, job_id: str) -> dict[str, Any]:
    raw = conn.get(_RESPONSES.format(j=job_id))
    if not raw:
        raise RuntimeError(f"No responses in Redis for job {job_id}")
    return _decompress_json(raw)


# ── Vectors ──────────────────────────────────────────────────────────

def store_vectors(conn: Redis, job_id: str, vectors: dict[str, np.ndarray]) -> None:
    conn.setex(_VECTORS.format(j=job_id), _ttl(), _compress_pickle(vectors))
    logger.info("stored_vectors", job_id=job_id, questions=list(vectors.keys()))


def load_vectors(conn: Redis, job_id: str) -> dict[str, np.ndarray]:
    raw = conn.get(_VECTORS.format(j=job_id))
    if not raw:
        raise RuntimeError(f"No vectors in Redis for job {job_id}")
    return _decompress_pickle(raw)


# ── Clusters ─────────────────────────────────────────────────────────

def store_clusters(conn: Redis, job_id: str, data: list[dict]) -> None:
    conn.setex(_CLUSTERS.format(j=job_id), _ttl(), _compress_json(data))
    logger.info("stored_clusters", job_id=job_id, count=len(data))


def load_clusters(conn: Redis, job_id: str) -> list[dict]:
    raw = conn.get(_CLUSTERS.format(j=job_id))
    if not raw:
        raise RuntimeError(f"No clusters in Redis for job {job_id}")
    return _decompress_json(raw)


# ── Analysis ─────────────────────────────────────────────────────────

def store_analysis(conn: Redis, job_id: str, data: list[dict]) -> None:
    conn.setex(_ANALYSIS.format(j=job_id), _ttl(), _compress_json(data))
    logger.info("stored_analysis", job_id=job_id, count=len(data))


def load_analysis(conn: Redis, job_id: str) -> list[dict]:
    raw = conn.get(_ANALYSIS.format(j=job_id))
    if not raw:
        raise RuntimeError(f"No analysis in Redis for job {job_id}")
    return _decompress_json(raw)


# ── Job State ────────────────────────────────────────────────────────

def update_state(
    conn: Redis,
    job_id: str,
    status: str,
    progress: float,
    message: str = "",
    error: str = "",
    report_id: str = "",
) -> None:
    from datetime import datetime, timezone
    key = _STATE.format(j=job_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.hset(key, "status", status)
    conn.hset(key, "progress", str(progress))
    conn.hset(key, "message", message)
    conn.hset(key, "error", error)
    conn.hset(key, "report_id", report_id)
    conn.hset(key, "updated_at", now_iso)
    conn.expire(key, _ttl())


# ── Cleanup ───────────────────────────────────────────────────────────

def cleanup_temp_data(conn: Redis, job_id: str) -> None:
    """Deletes all intermediate data. State key is kept for status polling."""
    for template in [_DATA, _RESPONSES, _VECTORS, _CLUSTERS, _ANALYSIS]:
        conn.delete(template.format(j=job_id))
    logger.info("redis_cleanup_done", job_id=job_id)


def record_metric(conn: Redis, job_id: str, name: str, value: float | int) -> None:
    key = _METRICS.format(j=job_id)
    conn.hset(key, name, str(value))
    conn.expire(key, _ttl())


def incr_metric(conn: Redis, job_id: str, name: str, amount: int = 1) -> None:
    key = _METRICS.format(j=job_id)
    conn.hincrby(key, name, amount)
    conn.expire(key, _ttl())


def mark_stage_completed(conn: Redis, job_id: str, stage_name: str) -> None:
    key = _STAGES.format(j=job_id)
    conn.hset(key, stage_name, "done")
    conn.expire(key, _ttl())


def is_stage_completed(conn: Redis, job_id: str, stage_name: str) -> bool:
    key = _STAGES.format(j=job_id)
    return bool(conn.hexists(key, stage_name))
