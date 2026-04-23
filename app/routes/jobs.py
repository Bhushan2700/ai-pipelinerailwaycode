from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from redis import Redis
from rq import Queue
from app.dependencies import get_pipeline_queue, get_redis_connection
from models.jobs import JobRequest, JobStatus, JobStatusResponse
from workers.tasks import run_survey_pipeline
import uuid

router = APIRouter()


@router.post("/", response_model=JobStatusResponse, status_code=202)
async def submit_job(
    request: JobRequest,
    queue: Queue = Depends(get_pipeline_queue),
    conn: Redis = Depends(get_redis_connection),
) -> JobStatusResponse:
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    state_key = f"job:{job_id}:state"
    conn.hset(
        state_key,
        mapping={
            "status": JobStatus.QUEUED.value,
            "progress": "0.0",
            "message": "Job queued",
            "report_id": "",
            "error": "",
            "survey_id": request.survey_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        },
    )
    from core.config import get_settings
    conn.expire(state_key, get_settings().redis.result_ttl)

    queue.enqueue(
        run_survey_pipeline,
        job_id,
        request.survey_id,
        request.layout,
        request.options,
        job_id=job_id,
    )

    return JobStatusResponse(
        job_id=job_id,
        survey_id=request.survey_id,
        status=JobStatus.QUEUED,
        progress=0.0,
        message="Job queued",
        created_at=now,
        updated_at=now,
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    conn: Redis = Depends(get_redis_connection),
) -> JobStatusResponse:
    state_key = f"job:{job_id}:state"
    data = conn.hgetall(state_key)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    def decode(v: bytes | str) -> str:
        return v.decode() if isinstance(v, bytes) else v

    return JobStatusResponse(
        job_id=job_id,
        survey_id=decode(data.get(b"survey_id", b"")),
        status=JobStatus(decode(data.get(b"status", b"queued"))),
        progress=float(decode(data.get(b"progress", b"0.0"))),
        message=decode(data.get(b"message", b"")),
        report_id=decode(data.get(b"report_id", b"")) or None,
        error=decode(data.get(b"error", b"")) or None,
        created_at=datetime.fromisoformat(decode(data.get(b"created_at", b"2000-01-01T00:00:00+00:00"))),
        updated_at=datetime.fromisoformat(decode(data.get(b"updated_at", b"2000-01-01T00:00:00+00:00"))),
    )
