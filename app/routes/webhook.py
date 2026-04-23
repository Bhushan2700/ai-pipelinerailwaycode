from fastapi import APIRouter, HTTPException, Header, Depends, Request
from pydantic import BaseModel
from core.config import get_settings
from core.exceptions import FileMakerAuthError, FileMakerFetchError
from core.logging import get_logger
from services.filemaker import FileMakerClient
import uuid

router = APIRouter()
logger = get_logger(__name__)


class FileMakerWebhookPayload(BaseModel):
    event_id: str
    record_id: str


async def verify_api_key(x_api_key: str = Header(default="")):
    """
    Verify that the X-API-Key header matches the configured webhook API key.
    - In production/staging: rejects if no key is configured (fail-closed).
    - In development: warns but allows through if no key is configured.
    """
    settings = get_settings()

    # If no API key is configured on the server side
    if not settings.webhook_api_key:
        if settings.environment not in ("development", "dev", "local"):
            logger.error("webhook_no_api_key_configured", environment=settings.environment)
            raise HTTPException(
                status_code=500,
                detail="Server misconfigured: no webhook API key set",
            )
        logger.warning("webhook_no_api_key_configured", hint="Set WEBHOOK_API_KEY in .env")
        return x_api_key

    # Validate the provided key
    if x_api_key != settings.webhook_api_key:
        logger.warning("webhook_unauthorized", reason="invalid_api_key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key

# //original
# @router.post("/")
# async def receive_webhook(
#     payload: FileMakerWebhookPayload,
#     background_tasks: BackgroundTasks,
#     request: Request,
# ):
#     """
#     Entry point called by FileMaker.
#     Returns 200 immediately so FileMaker doesn't time out.
#     Triggers the pipeline in the background.
#     """
#     job_id = str(uuid.uuid4())
#     client_ip = request.client.host if request.client else "unknown"
#     logger.info(
#         "webhook_received",
#         event_id=payload.event_id,
#         record_id=payload.record_id,
#         job_id=job_id,
#         client_ip=client_ip,
#     )

#     # Return immediately — FileMaker must not be kept waiting
#     background_tasks.add_task(_trigger_pipeline, job_id, payload.event_id, payload.record_id)

#     return {
#         "status": "received",
#         "job_id": job_id,
#         "event_id": payload.event_id,
#         "record_id": payload.record_id,
#         "message": "Pipeline triggered",
#     }
@router.post("/")
async def receive_webhook(
    request: Request,
    _: str = Depends(verify_api_key),
):
    import json

    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"

    from queues.connection import get_question_queue, get_redis_connection

    queue = get_question_queue()
    conn = get_redis_connection()
    queue_depth = queue.count
    hard_limit = settings.redis.queue_hard_limit or settings.redis.max_queue_depth
    soft_limit = settings.redis.queue_soft_limit
    if queue_depth >= hard_limit:
        logger.warning(
            "webhook_rejected_overloaded",
            queue_depth=queue_depth,
            queue_hard_limit=hard_limit,
            client_ip=client_ip,
        )
        raise HTTPException(
            status_code=503,
            detail="System overloaded, try again shortly",
        )
    if queue_depth >= soft_limit:
        logger.warning(
            "webhook_queue_pressure",
            queue_depth=queue_depth,
            queue_soft_limit=soft_limit,
            client_ip=client_ip,
        )

    # Parse JSON safely
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # FileMaker sometimes sends a JSON string body
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed JSON string")

    # Validate payload
    try:
        payload = FileMakerWebhookPayload(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation failed: {str(e)}")

    # Idempotency key prevents duplicate full pipeline runs on webhook retries.
    dedup_key = f"webhook:{payload.event_id}:{payload.record_id}"
    existing_job = conn.get(dedup_key)
    if existing_job:
        existing_job_id = existing_job.decode() if isinstance(existing_job, bytes) else str(existing_job)
        state = conn.hgetall(f"job:{existing_job_id}:state")
        decoded_state = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else str(v)
            )
            for k, v in state.items()
        } if state else {}
        existing_status = decoded_state.get("status", "")

        # Allow replay if previous job has already terminally finished.
        if existing_status in {"failed", "complete"}:
            conn.delete(dedup_key)
            logger.info(
                "webhook_dedup_released_for_terminal_job",
                event_id=payload.event_id,
                record_id=payload.record_id,
                previous_job_id=existing_job_id,
                previous_status=existing_status,
            )
        else:
            logger.info(
                "webhook_duplicate_suppressed",
                event_id=payload.event_id,
                record_id=payload.record_id,
                job_id=existing_job_id,
                existing_status=existing_status or "unknown",
                client_ip=client_ip,
            )
            return {
                "status": "duplicate",
                "job_id": existing_job_id,
                "event_id": payload.event_id,
                "record_id": payload.record_id,
                "message": "Duplicate webhook suppressed",
            }

    job_id = str(uuid.uuid4())
    dedup_set = conn.set(dedup_key, job_id, nx=True, ex=settings.redis.result_ttl)
    if not dedup_set:
        existing_job = conn.get(dedup_key)
        existing_job_id = (
            existing_job.decode() if isinstance(existing_job, bytes) else str(existing_job)
        ) if existing_job else ""
        return {
            "status": "duplicate",
            "job_id": existing_job_id,
            "event_id": payload.event_id,
            "record_id": payload.record_id,
            "message": "Duplicate webhook suppressed",
        }

    logger.info(
        "webhook_received",
        event_id=payload.event_id,
        record_id=payload.record_id,
        job_id=job_id,
        client_ip=client_ip,
        queue_depth=queue_depth,
    )

    # Durable enqueue before response prevents accept-but-lost window.
    try:
        _trigger_pipeline(job_id, payload.event_id, payload.record_id)
    except Exception as exc:
        conn.delete(dedup_key)
        logger.error("webhook_enqueue_failed", job_id=job_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Unable to enqueue pipeline job") from exc

    return {
        "status": "accepted",
        "job_id": job_id,
        "event_id": payload.event_id,
        "record_id": payload.record_id,
        "message": "Pipeline triggered",
    }

@router.post("/test-fetch")
async def test_fetch(payload: FileMakerWebhookPayload):
    """
    Test endpoint — receives event_id + record_id and immediately
    fetches matching records from N8N_SURVEY_EVENTS.
    Use this to verify the FileMaker → App connection is working
    before enabling the full pipeline.
    """
    settings = get_settings()
    logger.info("test_fetch_start", event_id=payload.event_id)

    try:
        async with FileMakerClient(settings.filemaker) as fm:
            records = await fm.fetch_event_responses(payload.event_id)
    except FileMakerAuthError as exc:
        raise HTTPException(status_code=503, detail=f"FileMaker auth failed: {exc}")
    except FileMakerFetchError as exc:
        raise HTTPException(status_code=502, detail=f"FileMaker fetch failed: {exc}")

    if not records:
        return {
            "status": "no_records",
            "event_id": payload.event_id,
            "message": "No records found for this event_id in N8N_SURVEY_EVENTS",
        }

    sample = records[0].get("fieldData", {})
    return {
        "status": "success",
        "event_id": payload.event_id,
        "total_records": len(records),
        "fields_detected": list(sample.keys()),
        "sample_record": sample,
    }


def _trigger_pipeline(job_id: str, event_id: str, record_id: str) -> None:
    from datetime import datetime, timezone
    from queues.connection import get_redis_connection, get_question_queue
    from core.config import get_settings
    from workers.tasks import fetch_data_task
    from rq import Retry

    settings = get_settings()
    conn = get_redis_connection()
    now = datetime.now(timezone.utc).isoformat()

    # Initialise job state in Redis
    conn.hset(f"job:{job_id}:state", mapping={
        "status":     "queued",
        "progress":   "0.0",
        "message":    "Job queued",
        "error":      "",
        "report_id":  "",
        "event_id":   event_id,
        "record_id":  record_id,
        "created_at": now,
        "updated_at": now,
    })
    conn.hset(f"job:{job_id}:metrics", mapping={"queued_at": now})
    conn.expire(f"job:{job_id}:metrics", settings.redis.result_ttl)
    conn.expire(f"job:{job_id}:state", settings.redis.result_ttl)

    queue = get_question_queue()
    queue.enqueue(
        fetch_data_task,
        job_id,
        event_id,
        record_id,
        job_timeout=settings.redis.job_timeout,
        retry=Retry(max=settings.redis.job_retry_max, interval=settings.redis.retry_backoff_seconds),
    )
    logger.info("pipeline_enqueued", job_id=job_id, event_id=event_id)
