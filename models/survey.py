from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class QuestionType(str, Enum):
    OPEN_ENDED = "open_ended"
    LIKERT = "likert"
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"


class SurveyResponse(BaseModel):
    record_id: str
    respondent_id: str
    question_id: str
    question_text: str
    question_type: QuestionType
    response_text: Optional[str] = None
    response_value: Optional[Any] = None
    metadata: dict[str, Any] = {}


class QuestionGroup(BaseModel):
    question_id: str
    question_text: str
    question_type: QuestionType
    responses: list[SurveyResponse]


class SurveyData(BaseModel):
    survey_id: str
    layout: str
    open_ended_questions: list[QuestionGroup]
    structured_questions: list[QuestionGroup]
    total_responses: int
