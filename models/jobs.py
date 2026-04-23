from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


class JobStatus(str, Enum):
    QUEUED = "queued"
    FETCHING = "fetching"
    VECTORIZING = "vectorizing"
    CLUSTERING = "clustering"
    ANALYZING = "analyzing"
    GENERATING_REPORT = "generating_report"
    STORING = "storing"
    COMPLETE = "complete"
    FAILED = "failed"


class JobRequest(BaseModel):
    survey_id: str
    layout: str
    options: dict[str, Any] = {}


class JobStatusResponse(BaseModel):
    job_id: str
    survey_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    message: str = ""
    report_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
