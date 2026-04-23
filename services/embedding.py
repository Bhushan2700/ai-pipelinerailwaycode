import asyncio
from typing import Optional
import numpy as np
from openai import AsyncOpenAI
from core.config import OpenAISettings
from core.exceptions import EmbeddingBatchError, EmbeddingRateLimitError
from core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, settings: OpenAISettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.api_key)

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        size = batch_size or self._settings.embedding_batch_size
        batches = self._split_into_batches(texts, size)
        logger.info("embedding_start", total=len(texts), batches=len(batches), batch_size=size)

        results = await self._process_batches_with_concurrency(batches)

        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)

        return np.array(all_embeddings, dtype=np.float32)

    def _split_into_batches(self, texts: list[str], batch_size: int) -> list[list[str]]:
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    async def _process_batches_with_concurrency(
        self,
        batches: list[list[str]],
        max_concurrent: int = 3,
    ) -> list[list[list[float]]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[Optional[list[list[float]]]] = [None] * len(batches)

        async def process(idx: int, batch: list[str]) -> None:
            async with semaphore:
                results[idx] = await self._embed_batch(batch)

        await asyncio.gather(*[process(i, b) for i, b in enumerate(batches)])
        return results  # type: ignore[return-value]

    async def _embed_batch(
        self,
        batch: list[str],
        attempt: int = 0,
    ) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._settings.embedding_model,
                input=batch,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            error_str = str(exc).lower()
            if "rate" in error_str or "429" in error_str:
                if attempt >= self._settings.max_retries:
                    raise EmbeddingRateLimitError(f"Rate limit exhausted after {attempt} retries") from exc
                wait = self._settings.retry_backoff_base ** attempt
                logger.warning("embedding_rate_limit", attempt=attempt, wait=wait)
                await asyncio.sleep(wait)
                return await self._embed_batch(batch, attempt + 1)
            if attempt >= self._settings.max_retries:
                raise EmbeddingBatchError(f"Batch embedding failed after {attempt} retries: {exc}") from exc
            wait = self._settings.retry_backoff_base ** attempt
            await asyncio.sleep(wait)
            return await self._embed_batch(batch, attempt + 1)
