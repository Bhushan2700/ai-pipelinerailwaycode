from fastapi import APIRouter, HTTPException
from core.config import get_settings
from core.exceptions import FileMakerAuthError, FileMakerFetchError
from core.logging import get_logger
from queues.connection import get_redis_connection
from rq import Queue
from services.filemaker import FileMakerClient

router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
async def health_check():
    """Basic health with queue pressure snapshot."""
    try:
        conn = get_redis_connection()
        conn.ping()
        queue_depths = {
            "question_processing": Queue("question_processing", connection=conn).count,
            "survey_pipeline": Queue("survey_pipeline", connection=conn).count,
            "default": Queue("default", connection=conn).count,
        }
        redis_info = conn.info("memory")
        used_memory_human = redis_info.get("used_memory_human", "unknown")
        redis_status = "connected"
    except Exception as exc:
        queue_depths = {}
        used_memory_human = "unknown"
        redis_status = f"failed: {exc}"

    return {
        "status": "ok" if redis_status == "connected" else "degraded",
        "redis": redis_status,
        "queue_depths": queue_depths,
        "redis_used_memory": used_memory_human,
    }


@router.get("/filemaker")
async def filemaker_health():
    """
    Tests FileMaker connection end-to-end:
    1. Authenticates and gets a session token
    2. Fetches layout list to confirm database access
    3. Releases the token
    """
    settings = get_settings()
    try:
        async with FileMakerClient(settings.filemaker) as fm:
            info = await fm.ping()
        return {
            "status": "connected",
            "host": settings.filemaker.host,
            "database": settings.filemaker.database,
            "layouts_accessible": info["layouts_accessible"],
        }
    except FileMakerAuthError as exc:
        return {
            "status": "auth_failed",
            "error": str(exc),
            "hint": "Check FM__USERNAME and FM__PASSWORD in your .env",
        }
    except FileMakerFetchError as exc:
        return {
            "status": "connection_failed",
            "error": str(exc),
            "hint": "Check FM__HOST and FM__DATABASE in your .env",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


@router.get("/job/{job_id}")
async def job_status(job_id: str):
    """Check the status of a pipeline job."""
    conn = get_redis_connection()
    data = conn.hgetall(f"job:{job_id}:state")
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in data.items()}


@router.get("/filemaker/layout")
async def filemaker_layout_test():
    """
    Tests that N8N_SURVEY_EVENTS layout is accessible.
    Returns first record structure so you can verify field names.
    """
    settings = get_settings()
    try:
        async with FileMakerClient(settings.filemaker) as fm:
            resp = await fm.find_records(
                layout="N8N_SURVEY_EVENTS",
                query=[{"event_id": "*"}],   # wildcard — matches any value
                limit=1,
            )
        if not resp:
            return {
                "status": "layout_accessible",
                "records_found": 0,
                "note": "Layout exists but has no records yet",
            }
        sample = resp[0].get("fieldData", {})
        return {
            "status": "layout_accessible",
            "records_found": len(resp),
            "fields_detected": list(sample.keys()),
            "sample_record": sample,
        }
    except Exception as exc:
        return {
            "status": "layout_error",
            "error": str(exc),
            "hint": "Verify the layout name N8N_SURVEY_EVENTS exists in FileMaker",
        }
