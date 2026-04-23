import numpy as np
import pytest
from core.config import ClusteringSettings
from services.clustering import ClusteringService
from models.survey import QuestionType, SurveyResponse


@pytest.fixture
def settings():
    return ClusteringSettings(min_cluster_size=2, min_samples=1)


@pytest.fixture
def service(settings):
    return ClusteringService(settings)


@pytest.fixture
def responses():
    return [
        SurveyResponse(
            record_id=str(i),
            respondent_id=f"R{i}",
            question_id="Q1",
            question_text="How was it?",
            question_type=QuestionType.OPEN_ENDED,
            response_text=f"Response {i}",
        )
        for i in range(6)
    ]


@pytest.fixture
def two_cluster_embeddings():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[1.0] * 8, scale=0.05, size=(3, 8)).astype(np.float32)
    b = rng.normal(loc=[-1.0] * 8, scale=0.05, size=(3, 8)).astype(np.float32)
    return np.vstack([a, b])


def test_cluster_question_returns_result(service, responses, two_cluster_embeddings):
    result = service.cluster_question("Q1", responses, two_cluster_embeddings)
    assert result.question_id == "Q1"
    assert result.total_responses == 6
    assert len(result.cluster_labels) == 6


def test_cluster_labels_have_correct_count(service, responses, two_cluster_embeddings):
    result = service.cluster_question("Q1", responses, two_cluster_embeddings)
    assert len(result.cluster_labels) == len(responses)


def test_noise_count_is_non_negative(service, responses, two_cluster_embeddings):
    result = service.cluster_question("Q1", responses, two_cluster_embeddings)
    assert result.noise_count >= 0


def test_raises_on_shape_mismatch(service, responses):
    wrong_embeddings = np.zeros((3, 8), dtype=np.float32)
    from core.exceptions import ClusteringError
    with pytest.raises(ClusteringError):
        service.cluster_question("Q1", responses, wrong_embeddings)
