import numpy as np
import pytest
from unittest.mock import AsyncMock, patch
from core.config import OpenAISettings
from services.embedding import EmbeddingService


@pytest.fixture
def settings():
    return OpenAISettings(api_key="test", embedding_batch_size=3, max_retries=2)


@pytest.fixture
def service(settings):
    return EmbeddingService(settings)


@pytest.mark.asyncio
async def test_embed_texts_respects_batch_size(service):
    texts = [f"text {i}" for i in range(10)]
    batch_calls: list[list[str]] = []

    async def fake_embed_batch(batch, attempt=0):
        batch_calls.append(batch)
        return [[0.1] * 8 for _ in batch]

    service._embed_batch = fake_embed_batch
    result = await service.embed_texts(texts, batch_size=3)

    assert result.shape[0] == 10
    assert all(len(b) <= 3 for b in batch_calls), "Batch size exceeded"


@pytest.mark.asyncio
async def test_embed_texts_preserves_order(service):
    texts = ["alpha", "beta", "gamma"]
    call_order: list[str] = []

    async def fake_embed_batch(batch, attempt=0):
        call_order.extend(batch)
        return [([float(i)] * 8) for i in range(len(batch))]

    service._embed_batch = fake_embed_batch
    result = await service.embed_texts(texts, batch_size=2)

    assert result.shape[0] == 3
    assert call_order[:2] == ["alpha", "beta"]
    assert call_order[2] == "gamma"


@pytest.mark.asyncio
async def test_empty_texts_returns_empty_array(service):
    result = await service.embed_texts([])
    assert result.shape[0] == 0


def test_split_into_batches(service):
    batches = service._split_into_batches(list(range(7)), 3)
    assert len(batches) == 3
    assert batches[0] == [0, 1, 2]
    assert batches[2] == [6]
