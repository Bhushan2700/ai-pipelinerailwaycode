from fastapi import APIRouter, Depends, HTTPException
from redis import Redis
from app.dependencies import get_redis_connection
from models.report import SurveyReport

router = APIRouter()

_REPORT_KEY = "report:{report_id}"


@router.get("/{report_id}", response_model=SurveyReport)
async def get_report(
    report_id: str,
    conn: Redis = Depends(get_redis_connection),
) -> SurveyReport:
    cached = conn.get(f"report:{report_id}")
    if cached:
        return SurveyReport.model_validate_json(cached)

    # Fallback: fetch from FileMaker
    from core.config import get_settings
    from services.filemaker import FileMakerClient

    settings = get_settings()
    try:
        async with FileMakerClient(settings.filemaker) as fm:
            records = await fm.find_records(
                settings.filemaker.layout_reports,
                [{"report_id": report_id}],
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not records:
        raise HTTPException(status_code=404, detail="Report not found")

    field_data = records[0].get("fieldData", {})
    report_json = field_data.get("report_json", "")
    if not report_json:
        raise HTTPException(status_code=404, detail="Report data missing")

    report = SurveyReport.model_validate_json(report_json)
    conn.setex(f"report:{report_id}", settings.redis.result_ttl, report_json)
    return report
