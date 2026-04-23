import json
import uuid
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any
from core.exceptions import ReportError
from core.logging import get_logger
from models.clusters import QuestionClusterResult
from models.report import OpenEndedQuestionReport, StructuredQuestionReport, SurveyReport
from models.survey import QuestionGroup, QuestionType

logger = get_logger(__name__)


class ReportService:
    def assemble_report(
        self,
        survey_id: str,
        open_ended_results: list[QuestionClusterResult],
        structured_questions: list[QuestionGroup],
    ) -> SurveyReport:
        open_sections = [
            OpenEndedQuestionReport(
                question_id=r.question_id,
                question_text=r.question_text,
                cluster_result=r,
                noise_percentage=round(r.noise_count / r.total_responses * 100, 2)
                if r.total_responses
                else 0.0,
            )
            for r in open_ended_results
        ]

        structured_sections = [
            self._aggregate_structured_question(q) for q in structured_questions
        ]

        return SurveyReport(
            report_id=str(uuid.uuid4()),
            survey_id=survey_id,
            generated_at=datetime.now(timezone.utc),
            open_ended_sections=open_sections,
            structured_sections=structured_sections,
        )

    def _aggregate_structured_question(self, question: QuestionGroup) -> StructuredQuestionReport:
        try:
            data = self._compute_aggregation(question)
        except Exception as exc:
            raise ReportError(f"Aggregation failed for {question.question_id}: {exc}") from exc
        return StructuredQuestionReport(
            question_id=question.question_id,
            question_text=question.question_text,
            question_type=question.question_type,
            aggregated_data=data,
        )

    def _compute_aggregation(self, question: QuestionGroup) -> dict[str, Any]:
        values = [r.response_value for r in question.responses if r.response_value is not None]
        if not values:
            return {"count": 0}

        if question.question_type == QuestionType.LIKERT:
            numeric = [float(v) for v in values]
            return {
                "count": len(numeric),
                "mean": round(mean(numeric), 2),
                "std_dev": round(stdev(numeric), 2) if len(numeric) > 1 else 0.0,
                "distribution": {
                    str(k): numeric.count(k) for k in sorted(set(numeric))
                },
            }

        if question.question_type == QuestionType.YES_NO:
            yes = sum(1 for v in values if str(v).lower() in ("yes", "true", "1"))
            return {
                "count": len(values),
                "yes_count": yes,
                "no_count": len(values) - yes,
                "yes_percentage": round(yes / len(values) * 100, 2),
            }

        # MULTIPLE_CHOICE or fallback
        from collections import Counter
        counts = Counter(str(v) for v in values)
        total = len(values)
        return {
            "count": total,
            "frequencies": dict(counts),
            "percentages": {k: round(v / total * 100, 2) for k, v in counts.items()},
        }

    def to_json(self, report: SurveyReport) -> str:
        return report.model_dump_json(indent=2)

    def to_filemaker_payload(self, report: SurveyReport) -> dict[str, Any]:
        return {
            "report_id": report.report_id,
            "survey_id": report.survey_id,
            "generated_at": report.generated_at.isoformat(),
            "executive_summary": report.executive_summary,
            "report_json": self.to_json(report),
        }
