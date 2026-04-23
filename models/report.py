from datetime import datetime
from typing import Any
from pydantic import BaseModel
from models.survey import QuestionType
from models.clusters import QuestionClusterResult


class StructuredQuestionReport(BaseModel):
    question_id: str
    question_text: str
    question_type: QuestionType
    aggregated_data: dict[str, Any]


class OpenEndedQuestionReport(BaseModel):
    question_id: str
    question_text: str
    cluster_result: QuestionClusterResult
    noise_percentage: float


class SurveyReport(BaseModel):
    report_id: str
    survey_id: str
    generated_at: datetime
    open_ended_sections: list[OpenEndedQuestionReport]
    structured_sections: list[StructuredQuestionReport]
    executive_summary: str = ""
    metadata: dict[str, Any] = {}
