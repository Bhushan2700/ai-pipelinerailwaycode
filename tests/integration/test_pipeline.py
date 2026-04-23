import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from models.survey import QuestionType, SurveyResponse
from models.clusters import ClusterAnalysis, ClusterLabel, QuestionClusterResult


@pytest.mark.asyncio
async def test_embedding_then_clustering_pipeline(sample_responses, deterministic_embeddings):
    from core.config import ClusteringSettings
    from services.clustering import ClusteringService

    settings = ClusteringSettings(min_cluster_size=2, min_samples=1)
    service = ClusteringService(settings)
    result = service.cluster_question("Q1", sample_responses, deterministic_embeddings)

    assert result.total_responses == len(sample_responses)
    assert len(result.cluster_labels) == len(sample_responses)
    assert result.noise_count >= 0
