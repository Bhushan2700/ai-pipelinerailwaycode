import pytest
from models.clusters import ClusterAnalysis, ClusterLabel, QuestionClusterResult
from models.survey import QuestionGroup, QuestionType, SurveyResponse
from services.report import ReportService


@pytest.fixture
def service():
    return ReportService()


@pytest.fixture
def cluster_result():
    return QuestionClusterResult(
        question_id="Q1",
        question_text="How was your experience?",
        total_responses=10,
        noise_count=1,
        clusters=[
            ClusterAnalysis(
                cluster_id=0,
                theme="Good service",
                summary="Respondents praised the service.",
                insights=["Staff were helpful"],
                representative_quotes=["Great staff"],
                response_count=5,
                percentage_of_total=50.0,
            )
        ],
        cluster_labels=[
            ClusterLabel(response_record_id=str(i), cluster_id=0, membership_probability=0.9)
            for i in range(9)
        ] + [ClusterLabel(response_record_id="9", cluster_id=-1, membership_probability=0.0)],
    )


@pytest.fixture
def structured_group():
    return QuestionGroup(
        question_id="Q2",
        question_text="Rate 1-5",
        question_type=QuestionType.LIKERT,
        responses=[
            SurveyResponse(
                record_id=str(i),
                respondent_id=f"R{i}",
                question_id="Q2",
                question_text="Rate 1-5",
                question_type=QuestionType.LIKERT,
                response_value=float(i % 5 + 1),
            )
            for i in range(10)
        ],
    )


def test_assemble_report_has_correct_ids(service, cluster_result, structured_group):
    report = service.assemble_report("survey-1", [cluster_result], [structured_group])
    assert report.survey_id == "survey-1"
    assert len(report.open_ended_sections) == 1
    assert len(report.structured_sections) == 1


def test_noise_percentage_calculated(service, cluster_result):
    report = service.assemble_report("s1", [cluster_result], [])
    section = report.open_ended_sections[0]
    assert section.noise_percentage == 10.0


def test_likert_aggregation(service, structured_group):
    agg = service._aggregate_structured_question(structured_group)
    assert "mean" in agg.aggregated_data
    assert "std_dev" in agg.aggregated_data
    assert agg.aggregated_data["count"] == 10


def test_to_json_roundtrip(service, cluster_result, structured_group):
    report = service.assemble_report("s1", [cluster_result], [structured_group])
    json_str = service.to_json(report)
    from models.report import SurveyReport
    restored = SurveyReport.model_validate_json(json_str)
    assert restored.survey_id == "s1"
