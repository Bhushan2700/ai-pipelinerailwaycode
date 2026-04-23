import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from models.survey import QuestionType, SurveyResponse


@pytest.fixture
def sample_responses() -> list[SurveyResponse]:
    texts = [
        "The service was excellent and staff were helpful",
        "Very good experience overall",
        "Staff helped me quickly",
        "Waiting time was too long",
        "Long queues at reception",
        "I waited over an hour",
        "Clean and well maintained facility",
        "The building was very clean",
        "Excellent cleanliness standards",
    ]
    return [
        SurveyResponse(
            record_id=str(i),
            respondent_id=f"R{i:03d}",
            question_id="Q1",
            question_text="How was your experience?",
            question_type=QuestionType.OPEN_ENDED,
            response_text=text,
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def deterministic_embeddings(sample_responses) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = len(sample_responses)
    embeddings = np.zeros((n, 16), dtype=np.float32)
    # Three clear clusters
    embeddings[:3] = rng.normal(loc=[1.0] * 16, scale=0.1, size=(3, 16))
    embeddings[3:6] = rng.normal(loc=[-1.0] * 16, scale=0.1, size=(3, 16))
    embeddings[6:] = rng.normal(loc=[0.0, 1.0] * 8, scale=0.1, size=(3, 16))
    return embeddings


@pytest.fixture
def mock_filemaker_client():
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.fetch_all_records = AsyncMock()
    client.create_record = AsyncMock(return_value="12345")
    client.bulk_create_records = AsyncMock(return_value=["1", "2"])
    return client


@pytest.fixture
def mock_openai_embeddings(mocker):
    async def fake_embed(texts, batch_size=None):
        rng = np.random.default_rng(sum(hash(t) for t in texts) % (2**31))
        return rng.random((len(texts), 1536)).astype(np.float32)

    mocker.patch(
        "services.embedding.EmbeddingService.embed_texts",
        side_effect=fake_embed,
    )
