import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.config import OpenAISettings
from models.clusters import ClusterAnalysis, QuestionClusterResult, ClusterLabel
from services.analysis import AnalysisService


@pytest.fixture
def settings():
    return OpenAISettings(api_key="test", completion_model="gpt-4o")


@pytest.fixture
def service(settings):
    return AnalysisService(settings)


@pytest.fixture
def cluster():
    return ClusterAnalysis(
        cluster_id=0,
        representative_quotes=["Great service", "Very helpful staff", "Excellent experience"],
        response_count=10,
        percentage_of_total=33.3,
    )


@pytest.mark.asyncio
async def test_analyze_cluster_fills_fields(service, cluster):
    fake_response = json.dumps({
        "theme": "Positive service experience",
        "summary": "Respondents appreciated the quality of service.",
        "insights": ["Staff training is effective", "Response times are fast"],
    })
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = fake_response

    with patch.object(service._client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)):
        result = await service.analyze_cluster(cluster, "How was the service?")

    assert result.theme == "Positive service experience"
    assert len(result.insights) == 2
    assert result.response_count == 10


@pytest.mark.asyncio
async def test_analyze_cluster_raises_on_bad_json(service, cluster):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "not json {"

    from core.exceptions import AnalysisError
    with patch.object(service._client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)):
        with pytest.raises(AnalysisError):
            await service.analyze_cluster(cluster, "Question?")
