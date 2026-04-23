import numpy as np
from core.config import ClusteringSettings
from core.exceptions import ClusteringError
from core.logging import get_logger
from models.clusters import ClusterAnalysis, ClusterLabel, QuestionClusterResult
from models.survey import SurveyResponse

logger = get_logger(__name__)

UMAP_THRESHOLD = 500
UMAP_N_COMPONENTS = 50


class ClusteringService:
    def __init__(self, settings: ClusteringSettings) -> None:
        self._settings = settings

    def cluster_question(
        self,
        question_id: str,
        responses: list[SurveyResponse],
        embeddings: np.ndarray,
    ) -> QuestionClusterResult:
        if len(responses) != embeddings.shape[0]:
            raise ClusteringError("responses and embeddings length mismatch")

        reduced = self._maybe_reduce_dimensions(embeddings)
        labels, probabilities = self._run_hdbscan(reduced)
        return self._build_cluster_result(question_id, responses, labels, probabilities)

    def _maybe_reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] <= UMAP_THRESHOLD:
            return embeddings
        try:
            from umap import UMAP
            reducer = UMAP(n_components=UMAP_N_COMPONENTS, metric="cosine", random_state=42)
            reduced = reducer.fit_transform(embeddings)
            logger.info("umap_reduction_applied", n=embeddings.shape[0], dims=UMAP_N_COMPONENTS)
            return reduced
        except ImportError:
            logger.warning("umap_not_available_skipping_reduction", n=embeddings.shape[0])
            return embeddings

    def _run_hdbscan(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = embeddings.shape[0]
        # HDBSCAN requires at least min_samples + 1 points to build k-NN graph.
        # Return all-noise arrays for datasets too small to cluster.
        if n <= self._settings.min_samples:
            logger.warning("dataset_too_small_for_clustering", n=n, min_samples=self._settings.min_samples)
            return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)

        try:
            import hdbscan

            # Normalize to unit length so euclidean distance == cosine distance.
            # The hdbscan BallTree backend doesn't support 'cosine' directly.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = embeddings / norms

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self._settings.min_cluster_size,
                min_samples=self._settings.min_samples,
                metric="euclidean",
                cluster_selection_method=self._settings.cluster_selection_method,
                prediction_data=True,
            )
            clusterer.fit(vectors)
            return clusterer.labels_, clusterer.probabilities_
        except Exception as exc:
            raise ClusteringError(f"HDBSCAN failed: {exc}") from exc

    def _build_cluster_result(
        self,
        question_id: str,
        responses: list[SurveyResponse],
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> QuestionClusterResult:
        total = len(responses)
        noise_count = int(np.sum(labels == -1))

        cluster_labels = [
            ClusterLabel(
                response_record_id=r.record_id,
                cluster_id=int(labels[i]),
                membership_probability=float(probabilities[i]),
            )
            for i, r in enumerate(responses)
        ]

        unique_ids = sorted(set(int(l) for l in labels if l != -1))
        clusters: list[ClusterAnalysis] = []

        for cid in unique_ids:
            mask = labels == cid
            cluster_responses = [r for r, m in zip(responses, mask) if m]
            cluster_probs = probabilities[mask]
            count = len(cluster_responses)

            # Pick top 3 by probability as representative quotes
            top_idx = np.argsort(cluster_probs)[::-1][:3]
            quotes = [
                cluster_responses[i].response_text or ""
                for i in top_idx
                if cluster_responses[i].response_text
            ]

            clusters.append(
                ClusterAnalysis(
                    cluster_id=cid,
                    representative_quotes=quotes,
                    response_count=count,
                    percentage_of_total=round(count / total * 100, 2),
                )
            )

        return QuestionClusterResult(
            question_id=question_id,
            question_text=responses[0].question_text if responses else "",
            total_responses=total,
            noise_count=noise_count,
            clusters=clusters,
            cluster_labels=cluster_labels,
        )
