class PipelineBaseError(Exception):
    pass


class FileMakerError(PipelineBaseError):
    pass


class FileMakerAuthError(FileMakerError):
    pass


class FileMakerFetchError(FileMakerError):
    pass


class FileMakerStoreError(FileMakerError):
    pass


class EmbeddingError(PipelineBaseError):
    pass


class EmbeddingBatchError(EmbeddingError):
    pass


class EmbeddingRateLimitError(EmbeddingError):
    pass


class ClusteringError(PipelineBaseError):
    pass


class AnalysisError(PipelineBaseError):
    pass


class ReportError(PipelineBaseError):
    pass
