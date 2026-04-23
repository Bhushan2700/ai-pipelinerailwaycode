from pydantic import BaseModel


class ClusterLabel(BaseModel):
    response_record_id: str
    cluster_id: int
    membership_probability: float


class ClusterAnalysis(BaseModel):
    cluster_id: int
    theme: str = ""
    summary: str = ""
    insights: list[str] = []
    representative_quotes: list[str] = []
    response_count: int = 0
    percentage_of_total: float = 0.0


class QuestionClusterResult(BaseModel):
    question_id: str
    question_text: str
    total_responses: int
    noise_count: int
    clusters: list[ClusterAnalysis]
    cluster_labels: list[ClusterLabel]
